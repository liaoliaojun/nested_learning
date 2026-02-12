# Nested Learning 复现实现

![Python](https://img.shields.io/badge/python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.9.0-red)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Status](https://img.shields.io/badge/tests-smoke--ready-lightgrey)

对 Google Nested Learning（HOPE）架构的机制级复现（HOPE blocks、CMS 与 Self‑Modifying TITANs），在保持完全开源与 `uv` 管理的前提下，达到 lucidrains TITAN 参考实现所设定的质量标准。

保真范围（高层）：
- ✅ HOPE / CMS / Self‑Modifying Titans 的更新规则与连线方式（机制级）
- ✅ 单元测试覆盖张量级不变量（teach-signal、δℓ、CMS chunking、causality）
- ⚠️ 在线“写入”为 stop-grad（不对在线更新 / boundary-state 训练过程反向传播）
- ⚠️ 本仓库不支持多 GPU 的“论文保真在线更新”（DDP 会禁用部分特性）

## 快速开始
```bash
uv python install 3.12
uv sync --all-extras
uv run bash scripts/data/run_sample.sh
uv run bash scripts/run_smoke.sh pilot  # CPU 友好的 HOPE block 冒烟测试
uv run bash scripts/run_e2e_smoke.sh    # sync + 样例数据 + 冒烟训练 + zeroshot 评测
uv run python scripts/eval/zeroshot.py \
  --config configs/hope/pilot.yaml \
  --checkpoint artifacts/examples/pilot_dummy.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks piqa --max-samples 32 --device cpu
```

## 环境要求
- Python 3.12+
- `uv` 包管理器（https://github.com/astral-sh/uv）
- PyTorch 2.9.0 LTS + 支持 CUDA 的 GPU（CPU 可用于冒烟测试）

## 安装
```bash
uv python install 3.12
uv sync --all-extras
```

开发检查项：
- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest`

## 数据流水线
1. **训练分词器**
   ```bash
   uv run python scripts/data/train_tokenizer.py \
     --manifest configs/data/refinedweb_mixture.yaml \
     --vocab-size 32000 \
     --output-dir artifacts/tokenizer/refinedweb_mix \
     --log-file data/mixtures/refinedweb_mix_tokenizer.json
   ```
2. **语料过滤 + 分片**
   ```bash
   uv run python scripts/data/process_mixture.py \
     configs/data/refinedweb_mixture_filtered.yaml \
     --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
     --log-file data/mixtures/refinedweb_mix_filtered_shards.json
   ```
3. **样例流水线**（下载/使用授权数据集、过滤、分片、记录统计）
   ```bash
   uv run bash scripts/data/run_sample.sh
   ```
4. **完整流水线**（通过 `RW_LIMIT`、`WIKI_LIMIT` 等环境变量扩展数据摄入规模）
   ```bash
   uv run bash scripts/data/run_full.sh  # 默认每个语料约 50k 文档；可按需提高上限
   ```

## 训练
- 单 GPU / CPU：
  ```bash
  uv run python train.py --config-name pilot_smoke
  ```
- Apple Silicon（若支持 MPS）：
  ```bash
  uv run python train.py --config-name pilot_smoke train.device=mps
  ```
- DDP（torchrun）：
  ```bash
  torchrun --nproc_per_node=2 train_dist.py --config-name mid
  ```
- 纯 CPU DDP 冒烟（验证 `gloo` 后端与确定性种子）：
  ```bash
  uv run bash scripts/run_cpu_ddp_smoke.sh
  ```
- FSDP（显存/批大小见 `docs/FSDP_SCALING_GUIDE.md`）：
  ```bash
  # 760M 训练
  torchrun --nproc_per_node=2 train_fsdp.py --config-name hope/mid_fsdp
  # 1.3B 训练
  torchrun --nproc_per_node=2 train_fsdp.py --config-name hope/target_fsdp
  ```
- DeepSpeed（需单独安装 `deepspeed`）：
  ```bash
  deepspeed --num_gpus=2 train_deepspeed.py --config-name target \
    deepspeed.config=configs/deepspeed/zero3.json
  ```

### 论文保真机制（HOPE / Nested Learning）

使用论文保真预设配置（单 GPU）：

```bash
uv run python train.py --config-name pilot_paper_faithful
# HOPE self-mod 变体：
uv run python train.py --config-name pilot_selfmod_paper_faithful
```

说明：
- 论文保真预设将 `data.batch_size=1`，以避免跨样本共享 fast-memory。

可覆盖项：
- `optim.type=m3`（论文优化器选项）
- `train.steps=...` / `train.device=...`

完整保真说明见 `docs/PAPER_COMPLIANCE.md`。

### Pilot（30 亿 tokens）工作流
1. 先启动 TMUX 会话：
   ```bash
   tmux new -s pilot_train
   ```
2. 在 `cuda:1` 上启动长程训练（约 52 小时墙钟时间）：
   ```bash
   set -a && source git.env && set +a
   export UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy
   uv run python train.py --config-name pilot \
     logging.enabled=true logging.backend=wandb \
     logging.project=nested-learning logging.run_name=pilot-main-$(date +%Y%m%d%H%M%S) \
     train.device=cuda:1
   ```
3. Checkpoint 每 1,000 步写入 `artifacts/checkpoints/pilot/step_*.pt`；对应的 W&B run 会记录完整遥测数据。
4. 将最终 checkpoint、配置、日志与评测 JSON/CSV 复制到 `artifacts/pilot_release/` 以便分发。

## 日志
在 Hydra 配置中设置 `logging.enabled=true`（或通过 CLI 覆盖）即可将指标发送到 W&B（默认）。如果要写本地 JSON 日志，使用 `logging.backend=json logging.path=logs/run.json`。样例输出位于 `logs/` 与 `artifacts/examples/`。

## 评测
- Zero-shot：
  ```bash
  uv run python scripts/eval/zeroshot.py \
  --config configs/hope/mid.yaml \
  --checkpoint checkpoints/mid/step_000100.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks all --max-samples 200 --device cuda:0
  ```
  使用 `uv run python scripts/eval/zeroshot.py --list-tasks` 显示完整基准任务列表（PIQA、HellaSwag、WinoGrande、ARC-E/C、BoolQ、SIQA、CommonsenseQA、OpenBookQA）。详情见 `docs/zeroshot_eval.md`。
- Needle-in-a-Haystack：
  ```bash
  uv run python scripts/eval/niah.py \
    --config configs/hope/mid.yaml \
    --checkpoint checkpoints/mid/step_000100.pt \
    --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
    --context-lengths 2048 4096 8192 --samples-per-length 20
  ```
- 持续学习遗忘评测：
  ```bash
  uv run python scripts/eval/continual.py \
    --config configs/hope/mid.yaml \
    --checkpoints checkpoints/mid/step_000050.pt checkpoints/mid/step_000100.pt \
    --segments-yaml configs/data/continual_segments_sample.yaml \
    --batch-size 4 --max-batches 10 --memorize --memorize-steps 2
  ```
  使用 `uv run python scripts/eval/plot_forgetting.py --continual-json eval/continual_mid.json` 绘制遗忘曲线。
- 长上下文诊断：
  ```bash
  uv run python scripts/eval/passkey.py --config configs/hope/pilot.yaml --checkpoint artifacts/checkpoints/pilot/step_230000.pt \
    --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --samples 64 --memorize

  uv run python scripts/eval/pg19_perplexity.py --config configs/hope/pilot.yaml --checkpoint artifacts/checkpoints/pilot/step_230000.pt \
    --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --max-samples 64
  ```

评测汇总会写入 `eval/`，并与各任务 JSON 指标文件并存。

### 测试时记忆（memorization）开关
所有评测器都支持 TITAN 风格记忆，以复现实测时自适应：
```bash
uv run python scripts/eval/zeroshot.py \
  ... \
  --memorize \
  --memorize-steps 2 \
  --memorize-use-correct-answer \
  --memorize-no-reset  # 可选：跨样本保留更新
  --memorize-paths titan,cms_fast \
  --memorize-surprise-threshold 0.01
```
- `--memorize` 打开学习器，默认每个样本执行 1 次 LMS 步。
- `--memorize-steps` 控制每个 prompt 的自适应轮数。
- `--memorize-use-correct-answer` 在记忆过程中注入真实答案文本（用于消融实验）。
- `--memorize-no-reset` 在样本间保留记忆；不加该参数则每题重置。
- `--memorize-paths` 限制接收 teach-signal 更新的层级（`titan`、`cms_fast` 或 `all`）。
- `--memorize-surprise-threshold` 基于 teach-signal 平均范数门控更新，对齐论文中的 surprise 触发机制。

记忆指标（baseline vs adaptive）会与任务准确率一同输出，便于对比。

## 架构变体
在 Hydra 配置中通过 `model.block_variant` 选择论文定义的变体：
- `hope_attention`（论文 HOPE-Attention）：`Attention → CMS`（论文定义）。
- `hope_selfmod`（论文 HOPE 骨架）：`Self-modifying Titans (Eqs. 83–93; Eq. 91 residual MLP memories) → CMS`，默认启用 **fixed q** 与 **local conv window=4**，并通过 `model.self_mod_chunk_size`（其他模块）与 `model.self_mod_chunk_size_memory`（M_memory）进行分块更新。“可微读取 / update-pass 写入”语义见 `docs/PAPER_COMPLIANCE.md`。
- `hope_hybrid`（历史版本）：`Attention + TitanMemory + CMS`（探索性，不是论文定义）。
- `transformer`（基线）：`Attention → MLP`（无 TITAN/CMS 学习更新；适合 Phase 2 对比）。

Self-modifying Titans 可调项（便于消融，且与论文对齐）：
- `model.self_mod_objective`（`l2` 或 `dot`）、`model.self_mod_use_rank1_precond`（类似 DGD 的预条件器）、`model.self_mod_use_alpha`（权重衰减/保留门）、`model.self_mod_stopgrad_vhat`、`model.self_mod_momentum`、`model.self_mod_adaptive_q`、`model.self_mod_local_conv_window`。

## Fast state（Nested Learning 语义）
上下文内更新可运行在每个上下文独立的 fast state 上，从而保证 meta 参数不会变化：
- `HOPEModel.init_fast_state()` / `TitanOnlyModel.init_fast_state()` 返回 `ModelFastState`。
- `MemorizeConfig.use_fast_state=true`（默认）要求把 `fast_state` 传入 `memorize_tokens()` / `memorize_sequence()`；评测脚本会自动处理。
- 训练同样可以在每个 batch 的 fast state 上执行更新 pass，配置为 `train.use_fast_state=true`（meta+delta fast state：meta 参数可学习，在线更新只写 delta）。若 `data.batch_size>1`，CMS/TITAN fast state 会在 batch 内共享；若需严格逐上下文语义请使用 `data.batch_size=1`。见 `docs/PAPER_COMPLIANCE.md`。

## 发布
在打 tag 或发布新 checkpoint 之前，请先执行 `docs/release_checklist.md`，确保发布包包含 manifest 校验报告、tokenizer coverage JSON、zero-shot/NIAH/continual/passkey/PG-19 评测输出、遗忘曲线图，以及填写完整的 checkpoint 报告。

## 性能与优化器选项
- **混合精度：** 通过 `train.mixed_precision.enabled=true train.mixed_precision.dtype=bf16` 启用 bf16 autocast（pilot/mid/target 配置默认已开启）。
- **`torch.compile`：** 通过 `train.compile.enable=true train.compile.mode=max-autotune` 加速 attention/core 循环；除非 `train.compile.strict=true`，否则失败会回退到 eager。
- **Muon 混合（默认）：** 所有 HOPE 配置默认 `optim.type=muon`，将 2D 及以上张量路由到 PyTorch 2.9 的 Muon 优化器，embeddings/norms 继续使用 AdamW。训练日志会输出 `optim.muon_param_elems` / `optim.adamw_param_elems` 便于确认拆分。
- **Fused AdamW 回退：** 若 Muon 不可用，或需对比 `reports/ablations.md` 中的 AdamW 消融，可覆盖为 `optim.type=adamw optim.fused=auto`。
- **Surprise 门控：** 设置 `model.surprise_threshold=<float>` 可门控所有内部更新。默认 surprise 指标为（缩放/裁剪后）teach signal 的平均 L2 范数（`model.surprise_metric=l2`）；消融时也可使用 `loss` 或 `logit_entropy`。评测 CLI 提供 `--memorize-surprise-threshold` 进行临时门控。

所有 Hydra 参数都可通过 CLI 覆盖，或通过配置组（`configs/hope/*.yaml`）组合。可与 `scripts/run_e2e_smoke.sh`（自动化）或 `scripts/run_cpu_ddp_smoke.sh`（纯 CPU 确定性检查）搭配，以快速验证发布版本。

## 文档与参考
- `docs/guide.md`：完整上手流程（安装 → 数据 → 训练 → 评测）。
- `docs/release_plan.md`：发布就绪检查清单。
- `docs/data_pipeline.md`：大规模分片/分词器工作流。
- `docs/scaling_guidance.md`：数据与算力扩展路线图。
- `docs/stage1_plan.md`、`docs/stage2_plan.md`：架构与实验路线图。
- `docs/stage2_progress.md`：最新双 GPU 训练/评测状态与命令。
- `docs/experiments_report.md`：已完成实验的论文草稿。
- `docs/stability_journal.md`：NaN 修复与 teach-scale 调优时间线记录。
- `docs/future_directions.md`：首版发布后的优先路线。
- `reports/stage2_smoke.md`：发布级冒烟流程的精确命令与产物。
- `docs/FSDP_SCALING_GUIDE.md`：双 RTX 6000 Ada 下 mid/target FSDP 配置说明。
- `google_papers/`：Nested Learning 与 TITAN 论文的 PDF/Markdown。
- `CHANGELOG.md`：各版本面向用户的变更记录。

## 贡献
1. 运行格式化/测试（`uv run ruff check .`、`uv run pytest`）。
2. 在 `docs/guide.md` 记录新增配置或脚本，并更新 `CHANGELOG.md`。
3. 提交 PR 时引用相关 NL/TITAN 规范章节或 planner transcript 片段。
