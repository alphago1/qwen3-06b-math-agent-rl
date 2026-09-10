import json
from pathlib import Path


ROOT = Path(__file__).parent / "results"


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


def paired(sft_name, grpo_name):
    sft = {x["source_id"]: x for x in load(sft_name)}
    grpo = {x["source_id"]: x for x in load(grpo_name)}
    ids = sorted(sft.keys() & grpo.keys())
    wins = [i for i in ids if not sft[i]["correct"] and grpo[i]["correct"]]
    losses = [i for i in ids if sft[i]["correct"] and not grpo[i]["correct"]]
    return {
        "n": len(ids),
        "sft_correct": sum(sft[i]["correct"] for i in ids),
        "grpo_correct": sum(grpo[i]["correct"] for i in ids),
        "sft_to_grpo_wins": len(wins),
        "sft_to_grpo_losses": len(losses),
        "both_right": sum(sft[i]["correct"] and grpo[i]["correct"] for i in ids),
        "both_wrong": sum(not sft[i]["correct"] and not grpo[i]["correct"] for i in ids),
        "win_examples": [{"sft": compact(sft[i]), "grpo": compact(grpo[i])} for i in wins[:3]],
        "loss_examples": [{"sft": compact(sft[i]), "grpo": compact(grpo[i])} for i in losses[:3]],
    }


result = {
    "run1_dev": paired("recheck_sft_dev.json", "grpo_dev.json"),
    "run1_test": paired("recheck_sft_test.json", "grpo_test.json"),
    "run2_test": paired("repeat2_sft_test.json", "repeat2_grpo_test.json"),
}
(ROOT / "grpo_paired_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: {x: y for x, y in v.items() if not x.endswith("examples")} for k, v in result.items()}, indent=2))
