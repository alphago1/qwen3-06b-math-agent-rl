import argparse
import ast
import json
import math
import operator
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_data import SYSTEM, TOOLS


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def calc(expr):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsafe expression")

    value = ev(ast.parse(expr, mode="eval"))
    if not math.isfinite(float(value)) or abs(float(value)) > 1e15:
        raise ValueError("non-finite or excessive result")
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def same_number(a, b):
    try:
        return abs(float(str(a).replace(",", "")) - float(str(b).replace(",", ""))) < 1e-8
    except Exception:
        return str(a).replace(",", "") == str(b).replace(",", "")


def load_rows(parquet_path, manifest_path, limit, seed):
    table = pq.read_table(parquet_path, columns=["question", "answer"])
    rows = table.to_pylist()
    ids = json.loads(Path(manifest_path).read_text())["train_source_ids"]
    selected = []
    for source_id in ids:
        match = FINAL_RE.search(rows[source_id]["answer"])
        if match:
            selected.append({"source_id": source_id, "question": rows[source_id]["question"], "answer": match.group(1).replace(",", "")})
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def prompt_text(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def rollout_group(model, tokenizer, row, group_size, max_turns, max_new_tokens, temperature):
    trajectories = []
    for _ in range(group_size):
        trajectories.append({
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}],
            "segments": [], "calls": [], "invalid": 0, "done": False, "final": "",
        })

    model.eval()
    for _ in range(max_turns):
        active = [i for i, trajectory in enumerate(trajectories) if not trajectory["done"]]
        if not active:
            break
        prompts = [prompt_text(tokenizer, trajectories[i]["messages"]) for i in active]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            **batch,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        offset = batch["input_ids"].shape[1]
        for batch_index, trajectory_index in enumerate(active):
            trajectory = trajectories[trajectory_index]
            completion = tokenizer.decode(outputs[batch_index, offset:], skip_special_tokens=True).strip()
            trajectory["segments"].append((prompts[batch_index], completion))
            trajectory["messages"].append({"role": "assistant", "content": completion})
            tool_match = TOOL_RE.search(completion)
            if not tool_match:
                trajectory["final"] = completion
                trajectory["invalid"] += int("<tool_call>" in completion)
                trajectory["done"] = True
                continue
            try:
                payload = json.loads(tool_match.group(1))
                if payload.get("name") != "calculator":
                    raise ValueError("wrong tool")
                expression = payload["arguments"]["expression"]
                result = calc(expression)
                trajectory["calls"].append(expression)
                trajectory["messages"].append({"role": "tool", "content": result})
            except Exception:
                trajectory["invalid"] += 1
                trajectory["messages"].append({"role": "tool", "content": "ERROR: invalid expression"})
    for trajectory in trajectories:
        match = FINAL_RE.search(trajectory["final"])
        prediction = match.group(1).replace(",", "") if match else None
        exact = prediction is not None and same_number(prediction, row["answer"])
        trajectory["prediction"] = prediction
        trajectory["exact"] = exact
        # Route A: pure terminal reward. Formatting and tool behavior do not earn
        # partial credit, so a well-formed but wrong trajectory cannot be promoted.
        trajectory["reward"] = float(exact)
    return trajectories


def token_logps(model, tokenizer, segments, grad):
    values = []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for prompt, completion in segments:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not completion_ids:
                continue
            input_ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
            logits = model(input_ids=input_ids, use_cache=False).logits[0]
            start = len(prompt_ids) - 1
            chosen_logits = logits[start:start + len(completion_ids)]
            targets = input_ids[0, len(prompt_ids):]
            values.append(torch.log_softmax(chosen_logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1))
    if not values:
        return torch.zeros(1, device=model.device, requires_grad=grad)
    return torch.cat(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--problems", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--save-every", type=int, default=8)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def base_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
        )

    policy = PeftModel.from_pretrained(base_model(), args.sft_adapter, is_trainable=True)
    reference = PeftModel.from_pretrained(base_model(), args.sft_adapter, is_trainable=False)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    rows = load_rows(args.parquet, args.manifest, args.problems, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "train_log.jsonl"
    start_time = time.time()
    totals = {"groups": 0, "trajectories": 0, "exact": 0, "updates": 0}

    with log_path.open("w", encoding="utf-8") as log_file:
        for step, row in enumerate(rows, 1):
            trajectories = rollout_group(policy, tokenizer, row, args.group_size, args.max_turns, args.max_new_tokens, args.temperature)
            rewards = torch.tensor([x["reward"] for x in trajectories], device=policy.device)
            advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            kls = []
            if float(rewards.std(unbiased=False)) > 1e-6:
                policy.train()
                for advantage, trajectory in zip(advantages, trajectories):
                    policy_logps = token_logps(policy, tokenizer, trajectory["segments"], grad=True)
                    reference_logps = token_logps(reference, tokenizer, trajectory["segments"], grad=False)
                    n = min(policy_logps.numel(), reference_logps.numel())
                    policy_logps, reference_logps = policy_logps[:n], reference_logps[:n]
                    log_ratio = reference_logps - policy_logps
                    kl = (torch.exp(log_ratio) - log_ratio - 1).mean()
                    loss = -advantage * policy_logps.mean() + args.beta * kl
                    losses.append(loss)
                    kls.append(kl.detach())
                group_loss = torch.stack(losses).mean()
                group_loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                totals["updates"] += 1
            else:
                group_loss = torch.tensor(0.0)

            record = {
                "step": step,
                "source_id": row["source_id"],
                "rewards": [round(x["reward"], 4) for x in trajectories],
                "exact": [x["exact"] for x in trajectories],
                "predictions": [x["prediction"] for x in trajectories],
                "calls": [x["calls"] for x in trajectories],
                "loss": float(group_loss.detach()),
                "kl": float(torch.stack(kls).mean()) if kls else 0.0,
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            totals["groups"] += 1
            totals["trajectories"] += len(trajectories)
            totals["exact"] += sum(x["exact"] for x in trajectories)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if step % args.save_every == 0:
                policy.save_pretrained(args.output / "adapter")

    policy.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    summary = {
        **totals,
        "sample_exact_rate": totals["exact"] / max(1, totals["trajectories"]),
        "wall_seconds": time.time() - start_time,
        "problems": len(rows),
        "group_size": args.group_size,
        "lr": args.lr,
        "beta": args.beta,
        "seed": args.seed,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
