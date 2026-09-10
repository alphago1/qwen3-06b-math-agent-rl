import argparse
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate one arithmetic expression exactly.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]
SYSTEM = (
    "Solve the grade-school math problem. Use the calculator tool for every "
    "arithmetic operation. Use only values supported by the question or prior "
    "tool results. End with the final numeric answer in the exact format "
    "#### number."
)
CALC_RE = re.compile(r"<<(.+?)=([^<>]+)>>")
FINAL_RE = re.compile(r"####\s*([^\s]+)")


def assistant_tool(expression: str):
    return {"role": "assistant", "tool_calls": [{
        "type": "function",
        "function": {"name": "calculator", "arguments": {"expression": expression}},
    }]}


def conversation(row):
    matches = CALC_RE.findall(row["answer"])
    final = FINAL_RE.search(row["answer"])
    if not matches or not final:
        return None
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}]
    for expr, result in matches:
        messages.append(assistant_tool(expr.strip()))
        messages.append({"role": "tool", "content": result.strip()})
    messages.append({"role": "assistant", "content": f"Using the verified calculations above, the answer is {final.group(1)}.\n#### {final.group(1)}"})
    return messages


def decision_rows(tokenizer, source_rows, split):
    rows=[]
    for source_id, row in source_rows:
        messages=conversation(row)
        if not messages:
            continue
        context=[]
        decision_index=0
        for message in messages:
            if message["role"] == "assistant":
                prompt=tokenizer.apply_chat_template(context, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                full=tokenizer.apply_chat_template(context+[message], tools=TOOLS, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                rows.append({"source_id":source_id,"split":split,"decision_index":decision_index,"prompt_text":prompt,"full_text":full})
                decision_index += 1
            context.append(message)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--raw-dir",type=Path)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--train",type=int,default=500)
    ap.add_argument("--dev",type=int,default=100)
    ap.add_argument("--test",type=int,default=200)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    if args.raw_dir:
        def read_jsonl(path):
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        train_path=args.raw_dir/"train.jsonl"; test_path=args.raw_dir/"test.jsonl"
        if train_path.exists():
            train=list(enumerate(read_jsonl(train_path)))
            test=list(enumerate(read_jsonl(test_path))) if test_path.exists() else []
        else:
            from datasets import load_dataset
            parquet=load_dataset("parquet",data_files={"train":str(args.raw_dir/"main/train-*.parquet"),"test":str(args.raw_dir/"main/test-*.parquet")})
            train=list(enumerate(parquet["train"])); test=list(enumerate(parquet["test"]))
    else:
        from datasets import load_dataset
        ds=load_dataset("openai/gsm8k","main")
        train=list(enumerate(ds["train"])); test=list(enumerate(ds["test"]))
    rng=random.Random(args.seed); rng.shuffle(train); rng.shuffle(test)
    selected_train=train[:args.train]; selected_dev=train[args.train:args.train+args.dev]
    selected_test=(test[:args.test] if test else train[args.train+args.dev:args.train+args.dev+args.test])
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=decision_rows(tokenizer,selected_train,"train")
    with (args.output_dir/"train.jsonl").open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    for name, data in [("dev",selected_dev),("test",selected_test)]:
        with (args.output_dir/f"{name}.jsonl").open("w",encoding="utf-8") as f:
            for source_id,row in data:
                final=FINAL_RE.search(row["answer"])
                f.write(json.dumps({"source_id":source_id,"question":row["question"],"answer":final.group(1) if final else None},ensure_ascii=False)+"\n")
    manifest={"seed":args.seed,"train_source_ids":[x[0] for x in selected_train],"dev_source_ids":[x[0] for x in selected_dev],"test_source_ids":[x[0] for x in selected_test],"train_decisions":len(rows),"tools":TOOLS,"system":SYSTEM}
    (args.output_dir/"split_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"train_decisions":len(rows),"dev":len(selected_dev),"test":len(selected_test)},indent=2))


if __name__ == "__main__": main()
