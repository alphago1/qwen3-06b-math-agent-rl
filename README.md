# Qwen3-0.6B 数学工具 Agent：GSM8K SFT→GRPO 推理 RL 最小闭环

在 **Qwen3-0.6B + 单卡** 上完整跑通"calculator 工具调用的 **决策级 SFT** → **纯终局奖励 GRPO**"两阶段推理 RL 实验。全部评测为**机械判分（不依赖 LLM judge）**，train/dev/test 以 seed=42 确定性划分，Toy Test 绝不进入训练。

## 实验结果

**阶段一：决策级 SFT（completion-only LoRA，单卡 198 秒）**

| 指标 | Base | SFT 后 |
|---|---|---|
| Toy Test 200 题严格准确率 | 1.5% | **16%**（+14.5pp） |
| 工具使用率 | 86.5% | **99%** |
| 格式合规率 | 7.5% | **97.5%** |

> 判分敏感性主动披露：宽松判分下 SFT 18.5% → 16.0% 略降；结论以严格判分为准。

**阶段二：纯终局奖励 GRPO 最小闭环（"无混合组不更新"纪律）**

| 指标 | SFT | GRPO 后 |
|---|---|---|
| Dev 准确率 | 19% | **23%** |
| Test 准确率 | 17% | **21.5%**（净增 9 题正确、仅 3 题退化） |

训练协议：首轮 64 题×4 候选，续训 3 轮每轮 128 题×4（train-only）；只有组内 reward 有差异（有对有错）的题组才产生更新。

> **数字出处（勿混轮次）**：表中 19%→23%、17%→21.5% 为 **round4 终评**（`results/multi_round/round4_sft_*` vs `round4_grpo_*`：test 34→43 正确，newly_correct 12 / regressed 3）；round1 GRPO test 为 20.0%（`results/grpo_test.summary.json`）；SFT test 严格率三次复测 16.0% / 17.0% / 17.5%，表内取 round4 口径 17.0%。

完整逐题结果与配对分析见 `results/`（`*.summary.json`、`grpo_paired_analysis.json`、`grpo_train_metrics.json`）。

> 定位声明：这是一个**教学向最小闭环**（toy experiment）——验证"可验证奖励 + 组相对优势"在小模型上各环节的真实有效性，不是多 epoch、大规模生产级 RL 训练，也不以刷 GSM8K SOTA 为目标。

## 方法要点

- **数据构造零 LLM 依赖**：直接从 GSM8K 答案标注（`<<48/2=24>>`）机械转换成 calculator 工具调用轨迹，只监督下一步动作（决策级 SFT），500 题展开 2,037 个决策样本，无需人工标注或 LLM 造数据；
- **自实现轨迹级 GRPO**：组内相对优势（无 critic）、assistant-token 掩码监督、纯终局奖励（答案对错可验证）；
- **训练/评测完全解耦**：vLLM 起服务并发评测，`evaluate_vllm.py` 支持多轮工具调用与自动判分。

## 目录结构

```
├── prepare_data.py          # GSM8K → calculator 工具调用轨迹（决策级 SFT 数据）
├── train_lora.py            # completion-only LoRA SFT（单卡 198 秒）
├── train_terminal_grpo.py   # 自实现轨迹级 GRPO（组归一优势、终局奖励）
├── evaluate_vllm.py         # vLLM 多轮工具调用评测 + 机械判分
├── analyze_grpo.py / analyze_results.py / analyze_round4.py   # 结果与配对分析
├── run_grpo_background.sh   # GRPO 训练启动脚本
├── requirements.txt
├── data/                    # train/dev/test jsonl（GSM8K 衍生，构造脚本见 prepare_data.py）
└── results/                 # 各阶段逐题结果、summary、配对分析、训练日志
```

## 复现

```bash
pip install -r requirements.txt
python prepare_data.py            # 生成 data/
python train_lora.py              # 阶段一 SFT
vllm serve <model> &              # 起评测服务
python evaluate_vllm.py --data data/dev.jsonl --model <sft_model>
bash run_grpo_background.sh       # 阶段二 GRPO
python analyze_grpo.py            # 结果分析
```

## License

[MIT](LICENSE)
