#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/math-toy
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/math_toy_interview/train_terminal_grpo.py"
EVAL="$ROOT/math_toy_interview/evaluate_vllm.py"
MODEL="$ROOT/models/Qwen3-0.6B"
PARQUET="$ROOT/raw_gsm8k_modelscope/main/train-00000-of-00001.parquet"
MANIFEST="$ROOT/data_modelscope/split_manifest.json"
STATUS="$ROOT/grpo_background_status.json"

write_status() {
  "$PY" - "$STATUS" "$1" "$2" <<'PY'
import json, sys, time
path, state, detail = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"state": state, "detail": detail, "updated_at": time.strftime("%F %T")}, handle, ensure_ascii=False, indent=2)
PY
}

trap 'write_status failed "pipeline stopped; inspect grpo_background.log and per-round logs"' ERR

run_round() {
  local name="$1"
  local adapter="$2"
  local seed="$3"
  local lr="$4"
  local beta="$5"
  write_status running "$name: 128 groups x 4 trajectories"
  "$PY" "$TRAIN" \
    --model "$MODEL" \
    --sft-adapter "$adapter" \
    --parquet "$PARQUET" \
    --manifest "$MANIFEST" \
    --output "$ROOT/$name" \
    --problems 128 \
    --group-size 4 \
    --lr "$lr" \
    --beta "$beta" \
    --seed "$seed" \
    --save-every 16 \
    > "$ROOT/$name.log" 2>&1
}

run_round grpo_round2 "$ROOT/grpo_terminal64/adapter" 2027 3e-6 0.03
run_round grpo_round3 "$ROOT/grpo_round2/adapter" 2028 2e-6 0.04
run_round grpo_round4 "$ROOT/grpo_round3/adapter" 2029 1e-6 0.05

write_status evaluating "starting vLLM and running one final SFT/GRPO dev+test comparison"
nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen3-0.6b-base \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.75 --max-num-seqs 64 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-lora --max-lora-rank 16 --max-loras 2 \
  --lora-modules \
    qwen3-0.6b-sft="$ROOT/sft/adapter" \
    qwen3-0.6b-grpo-r4="$ROOT/grpo_round4/adapter" \
  > "$ROOT/vllm_round4.log" 2>&1 < /dev/null &

for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 5
done
curl -sf http://127.0.0.1:8000/v1/models >/dev/null

eval_pids=()
"$PY" "$EVAL" --data "$ROOT/data_modelscope/dev.jsonl" --output "$ROOT/results/round4_sft_dev.json" --model qwen3-0.6b-sft --concurrency 16 > "$ROOT/round4_sft_dev.log" 2>&1 & eval_pids+=("$!")
"$PY" "$EVAL" --data "$ROOT/data_modelscope/test.jsonl" --output "$ROOT/results/round4_sft_test.json" --model qwen3-0.6b-sft --concurrency 16 > "$ROOT/round4_sft_test.log" 2>&1 & eval_pids+=("$!")
"$PY" "$EVAL" --data "$ROOT/data_modelscope/dev.jsonl" --output "$ROOT/results/round4_grpo_dev.json" --model qwen3-0.6b-grpo-r4 --concurrency 16 > "$ROOT/round4_grpo_dev.log" 2>&1 & eval_pids+=("$!")
"$PY" "$EVAL" --data "$ROOT/data_modelscope/test.jsonl" --output "$ROOT/results/round4_grpo_test.json" --model qwen3-0.6b-grpo-r4 --concurrency 16 > "$ROOT/round4_grpo_test.log" 2>&1 & eval_pids+=("$!")
for eval_pid in "${eval_pids[@]}"; do
  wait "$eval_pid"
done

write_status complete "three continuation rounds and final paired evaluation completed"
