import json
from pathlib import Path


ROOT = Path(__file__).parent / "results" / "multi_round"


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def compact(row):
    return {
        "source_id": row["source_id"],
        "question": row["question"],
        "gold": row["answer"],
        "prediction": row["prediction"],
        "tool_calls": row["tool_calls"],
        "final_text": row["final_text"],
    }


def paired(split):
    sft = {x["source_id"]: x for x in load(f"round4_sft_{split}.json")}
    grpo = {x["source_id"]: x for x in load(f"round4_grpo_{split}.json")}
    ids = sorted(sft.keys() & grpo.keys())
    wins = [i for i in ids if not sft[i]["correct"] and grpo[i]["correct"]]
    losses = [i for i in ids if sft[i]["correct"] and not grpo[i]["correct"]]
    return {
        "n": len(ids),
        "sft_correct": sum(sft[i]["correct"] for i in ids),
        "round4_correct": sum(grpo[i]["correct"] for i in ids),
        "newly_correct": len(wins),
        "regressed": len(losses),
        "net_gain": len(wins) - len(losses),
        "both_correct": sum(sft[i]["correct"] and grpo[i]["correct"] for i in ids),
        "both_wrong": sum(not sft[i]["correct"] and not grpo[i]["correct"] for i in ids),
        "win_examples": [{"sft": compact(sft[i]), "round4": compact(grpo[i])} for i in wins[:5]],
        "loss_examples": [{"sft": compact(sft[i]), "round4": compact(grpo[i])} for i in losses[:5]],
    }


result = {split: paired(split) for split in ("dev", "test")}
(ROOT / "round4_paired_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: {x: y for x, y in v.items() if not x.endswith("examples")} for k, v in result.items()}, indent=2))
