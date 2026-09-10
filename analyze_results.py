import json
from pathlib import Path


ROOT = Path(__file__).parent / "results"


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def short(row):
    return {
        "source_id": row["source_id"],
        "question": row["question"],
        "gold": row["answer"],
        "prediction": row["prediction"],
        "relaxed_prediction": row["relaxed_prediction"],
        "tool_calls": row["tool_calls"],
        "final_text": row["final_text"],
    }


def analyze(split):
    base = load(f"fixed_base_{split}.json")
    sft = load(f"fixed_sft_{split}.json")
    bmap = {x["source_id"]: x for x in base}
    smap = {x["source_id"]: x for x in sft}
    ids = sorted(bmap.keys() & smap.keys())
    out = {
        "n": len(ids),
        "base_format_rate": sum(bmap[i]["prediction"] is not None for i in ids) / len(ids),
        "sft_format_rate": sum(smap[i]["prediction"] is not None for i in ids) / len(ids),
        "base_avg_calls": sum(len(bmap[i]["tool_calls"]) for i in ids) / len(ids),
        "sft_avg_calls": sum(len(smap[i]["tool_calls"]) for i in ids) / len(ids),
        "strict_0_to_1": sum(not bmap[i]["correct"] and smap[i]["correct"] for i in ids),
        "strict_1_to_0": sum(bmap[i]["correct"] and not smap[i]["correct"] for i in ids),
        "strict_both_right": sum(bmap[i]["correct"] and smap[i]["correct"] for i in ids),
        "relaxed_0_to_1": sum(not bmap[i]["relaxed_correct"] and smap[i]["relaxed_correct"] for i in ids),
        "relaxed_1_to_0": sum(bmap[i]["relaxed_correct"] and not smap[i]["relaxed_correct"] for i in ids),
    }
    wins = [i for i in ids if not bmap[i]["correct"] and smap[i]["correct"]]
    losses = [i for i in ids if bmap[i]["relaxed_correct"] and not smap[i]["relaxed_correct"]]
    out["strict_win_examples"] = [{"base": short(bmap[i]), "sft": short(smap[i])} for i in wins[:3]]
    out["planning_loss_examples"] = [{"base": short(bmap[i]), "sft": short(smap[i])} for i in losses[:3]]
    return out


result = {split: analyze(split) for split in ("dev", "test")}
(ROOT / "paired_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: {x: y for x, y in v.items() if not x.endswith("examples")} for k, v in result.items()}, indent=2))
