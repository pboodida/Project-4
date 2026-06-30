# Project 4 Reverse Engineering Report: AutoGaze

*   **Project Name:** AutoGaze
*   **Repository:** [https://github.com/NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze)
*   **Project Category:** AI Models & Weights — Efficient Video Understanding
*   **Paper:** *Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing* — CVPR 2026
*   **arXiv:** [2603.12254](https://arxiv.org/abs/2603.12254)
*   **Deadline:** April 3rd, 2026

---

## 1. Project Overview and Key Components

### Repository Analysis Summary

AutoGaze is a CVPR 2026 accepted work from NVIDIA Research, UC Berkeley, and MIT that addresses the fundamental computational inefficiency of treating every video patch equally in Vision Transformers (ViTs) and Multimodal Large Language Models (MLLMs). Real-world video streams exhibit extreme spatiotemporal redundancy: a static background occupies 80%+ of frame area and changes across zero frames; only moving foreground objects and texture-rich regions carry new information from frame to frame. Despite this redundancy, all current video ViTs and MLLMs embed every patch of every frame into the same fixed-length token sequence, then process all tokens through attention layers with O(N²) complexity.

AutoGaze is a lightweight **~3M parameter autoregressive patch selector** that operates *before* any ViT or LLM in the pipeline. It processes each video frame sequentially, using its history of previously selected patches to decide which new patches to select, and outputs a minimal set of multi-scale patch indices that can reconstruct the original video within a user-specified quality threshold. Downstream ViTs and LLMs receive only these selected patches, reducing their input token count by **4×–100×** and achieving up to **19× end-to-end latency reduction** on video benchmarks. The system is designed as a drop-in module: adding AutoGaze to an existing ViT requires modifying only the patch embedding layer and the attention mask, leaving all encoder weights, MLP layers, and layer norms unchanged.

The training pipeline is two-stage: **Stage 1 (NTP)** trains AutoGaze with next-token prediction on algorithmically-generated greedy gazing sequences. **Stage 2 (GRPO)** refines the policy with reinforcement learning using reconstruction loss as the reward signal, enabling the model to discover gazing strategies that surpass the quality of greedy training labels.

---

## Key Components

### 1. `autogaze/models/autogaze/` — The Gaze Model

The gaze model is an autoregressive transformer decoder built on the same architectural principles as LLMs, but operating over a *patch vocabulary* instead of a word vocabulary. Key characteristics:

- **Vocabulary**: 265 tokens per frame across 4 spatial scales: 32×32, 64×64, 112×112, 224×224 pixels. This multi-scale vocabulary allows the model to select a coarse 32px patch for a large static region (cheap) or a precise 224px patch for a small high-detail region.
- **Autoregressive decoding**: At each step, the model predicts the next patch index conditioned on the full history of frames and previously selected patches—structurally identical to LLM text generation.
- **KV-cache support**: Supports streaming inference where one frame is processed at a time with cached attention states from prior frames, enabling real-time video processing.
- **Task loss prediction head**: A lightweight linear head on the decoder hidden state that predicts reconstruction loss at each selection step, enabling early stopping without invoking the expensive reconstruction oracle.

**Output keys:**

| Key | Shape | Description |
|---|---|---|
| `gazing_pos` | `(B, N)` | Patch indices selected (including padding) |
| `if_padded_gazing` | `(B, N)` | Boolean mask: True = padded dummy position |
| `num_gazing_each_frame` | `(T,)` | Patches selected per frame including padding |
| `log_action_probs` | `(B, N)` | Log-probability of each selected patch (for GRPO loss) |

### 2. `autogaze/tasks/video_mae_reconstruction/` — Task & Reward Definition

This module defines the training objective and reward signal for the gaze model. It contains:

- **VideoMAE reconstruction model**: A VideoMAE variant with block-causal attention used to reconstruct the full video from the selected patches. Frozen during AutoGaze training (`trainer.train_task=False`).
- **Multi-component loss**: `L = 1.0 × L1_pixel + 0.3 × DINOv2_features + 0.3 × SigLIP2_features`. The perceptual feature-level terms (DINOv2, SigLIP2) align the reconstruction objective with the feature spaces used by downstream classification models, enabling cross-task transfer.
- **Reward function**: Negative reconstruction loss (lower loss = higher reward), used as the GRPO reward signal.
- **Ground truth labels**: `gazing_labels.json` contains pre-computed greedy gazing sequences—the result of iteratively selecting the single patch that most reduces reconstruction loss at each step.

### 3. `autogaze/algorithms/grpo.py` & `ntp.py` — Training Algorithms

**NTP (`ntp.py`)**: Cross-entropy loss over predicted patch indices, supervised against `gazing_labels.json`. Includes auxiliary loss for task loss prediction head. Uses loss masking to ignore padded positions. Runs for 150 epochs with global batch size 1024.

**GRPO (`grpo.py`)**: Group Relative Policy Optimization with `group_size=12` rollouts per input, `discount_factor=0.995` temporal credit assignment, and within-group mean normalization for variance reduction. Runs for ~1 epoch (high per-step compute due to 12 VideoMAE forward passes per input). Key parameters:

```python
group_size = 12           # 12 gazing trajectories per video
discount_factor = 0.995   # near-undiscounted temporal credit
optimize_task_loss_prediction = True  # joint training of loss predictor
```

### 4. `autogaze/vision_encoders/siglip/` — ViT Adapter

A modified SigLIP implementation that accepts AutoGaze's `gazing_info` dict to process only selected patches. Two minimal code changes to the original HuggingFace SigLIP:

**Change 1 — `mask_with_gazing` in patch embedding:**
```python
def mask_with_gazing(self, sequence, gazing_info):
    gazing_pos = gazing_info['gazing_pos'].clone()
    if_padded_gazing = gazing_info['if_padded_gazing'].clone()
    B = sequence.shape[0]
    gazing_pos[if_padded_gazing] = 0  # dummy token for padded positions
    sequence_gazed = sequence[torch.arange(B)[:, None], gazing_pos]
    return sequence_gazed
```

**Change 2 — `get_causal_mask` for inter-frame attention:**
```python
def get_causal_mask(self, num_tokens_each_frame, batch_size, num_heads, ...):
    mask = torch.tril(torch.ones(batch_size, N, N))  # causal across frames
    for t in range(T):
        mask[:, start:end, start:end] = 1  # bidirectional within frame
    mask = mask * (~token_mask.unsqueeze(1))  # zero out padded tokens
    return torch.where(mask == 1, 0, -torch.inf)  # additive attention mask
```

All encoder layers, MLPs, and layer norms remain **unchanged**.

### 5. Training Data — `bfshi/AutoGaze-Training-Data`

Five video sub-datasets used for training:

| Dataset | Domain | Size |
|---|---|---|
| InternVid | General web video | 250K clips |
| 100DoH | Egocentric (hands) | 250K clips |
| Ego4D | Egocentric (diverse) | 250K clips |
| scanning_SAM | SA-1B video scans | 50K clips |
| scanning_idl | IDL document scans | 50K clips |

Plus `gazing_labels.json` with pre-computed greedy gazing sequences for 250K of these videos.

---

## System Architecture

### End-to-End Inference Pipeline

```
Input Video (any resolution, any duration)
         │
         ▼
┌────────────────────────────────┐
│   Spatial Tiling               │
│   1344×1344, 256 frames        │
│   → 16-frame 224×224 chunks    │
│   via einops.rearrange()       │
└─────────────┬──────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────--┐
│                   AUTOGAZE MODEL (3M params)             │
│                                                          │
│   ┌─────────────────┐         ┌──────────────────────┐   │
│   │  Conv Feature   │         │  Autoregressive      │   │
│   │  Extractor      │────────▶│  Transformer Decoder │   │
│   │  (per frame)    │         │  Vocab: 265 tokens   │   │
│   └─────────────────┘         │  (4 scales × patches)│   │
│                                │                      │  │
│   ┌──────────────────────────┐ │  Output per step:    │  │
│   │  KV-Cache (streaming)    │─│  - patch index       │  │
│   │  past_key_values         │ │  - log_action_prob   │  │
│   │  past_inputs_embeds      │ │  - predicted loss    │  │
│   │  past_attention_mask     │ └──────────────────────┘  │
│   │  past_conv_values        │            │              │
│   └──────────────────────────┘            │              │
│                                           ▼              │
│   ┌──────────────────────────────────────────────────-┐  │
│   │  Stopping Criterion:                              │  │
│   │  stop if predicted_loss < task_loss_requirement   │  │
│   │  OR patches_selected / total > gazing_ratio       │  │
│   └──────────────────────────────────────────────────-┘  │
└─────────────────────────┬──────────────────────────────--┘
                          │
              gazing_pos + if_padded_gazing
              num_gazing_each_frame
                          │
                          ▼
┌────────────────────────────────────────────────────────-┐
│                   SIGLIP ADAPTER                        │
│                                                         │
│   mask_with_gazing():                                   │
│     sequence[gazing_pos] → selected patch embeddings    │
│                                                         │
│   get_causal_mask():                                    │
│     block_causal: causal across frames,                 │
│                   bidirectional within frame            │
│     padded positions zeroed out                         │
│                                                         │
│   ViT encoder layers: UNCHANGED                         │
│   Output: (B, N, 768) features for N selected patches   │
└─────────────────────────┬──────────────────────────────-┘
                          │
          Filter: f[~if_padded_gazing]
                          │
                          ▼
┌────────────────────────────────────────────────────────-┐
│                   NVILA-8B LLM                          │
│   Input: visual tokens (213) + text tokens (~50)        │
│   4× to 100× fewer visual tokens vs. no AutoGaze        │
│   Output: video QA answer / caption                     │
└────────────────────────────────────────────────────────-┘
```

### Training Pipeline

```
┌──────────────────────────────────────────────────────────-┐
│                   STAGE 1: NTP PRE-TRAINING               │
│                                                           │
│  Dataset: 250K videos + gazing_labels.json                │
│  Algorithm: autogaze/algorithms/ntp.py                    │
│                                                           │
│  Loss:                                                    │
│    L_NTP = -Σ log P(patch_k | video, prior_patches)       │
│    + L2(predicted_recon_loss, actual_recon_loss)          │
│    masked at padded positions                             │
│                                                           │
│  Config: batch_size=1024, epochs=150, lr=5e-4             │
│  gazing_ratio=0.1 (10% of patches)                        │
│  optimize_task_loss_prediction=True                       │
└──────────────────────────────────────────────────────────-┘
                          │
                          ▼ NTP checkpoint
┌──────────────────────────────────────────────────────────-┐
│                   STAGE 2: GRPO RL                        │
│                                                           │
│  Dataset: same 800K videos (no gazing labels needed)      │
│  Algorithm: autogaze/algorithms/grpo.py                   │
│                                                           │
│  For each video:                                          │
│    1. Sample 12 gazing trajectories (group_size=12)       │
│    2. Run frozen VideoMAE → 12 reconstruction losses      │
│    3. Advantage_i = Loss_i - mean(Loss_1..Loss_12)        │
│    4. GRPO gradient: ∇ Σ log_prob_i × Advantage_i × γ^t   │
│                                                           │
│  Config: batch_size=64, epochs=1, lr=5e-4                 │
│  discount_factor=0.995, group_size=12                     │
│  gazing_ratio=0.75 (75% max per frame)                    │
└──────────────────────────────────────────────────────────-┘
                          │
                          ▼ Final model: bfshi/AutoGaze
```

---

## Repository File Map

```
NVlabs/AutoGaze/
├── README.md                     ← Project overview, installation, code structure
├── QUICK_START.md                ← Inference code: basic, advanced, streaming usage
├── TRAIN.md                      ← Training configuration and parameter documentation  
├── INTEGRATION.md                ← ViT adapter guide: mask_with_gazing, get_causal_mask
├── pyproject.toml                ← Package dependencies
│
├── autogaze/
│   ├── models/
│   │   └── autogaze/             ← Gaze model: autoregressive decoder + conv encoder
│   ├── algorithms/
│   │   ├── grpo.py               ← GRPO RL training logic
│   │   └── ntp.py                ← NTP supervised pre-training logic
│   ├── tasks/
│   │   └── video_mae_reconstruction/  ← VideoMAE task, reward function, gazing labels
│   ├── vision_encoders/
│   │   └── siglip/               ← Modified SigLIP: mask_with_gazing, get_causal_mask
│   ├── datasets/
│   │   └── video_folder.py       ← Video dataset loader
│   ├── configs/                  ← Hydra configs for model, task, algorithm, trainer
│   ├── trainer.py                ← Training loop: NTP or GRPO
│   └── train.py                  ← Entry point: instantiates model, task, algorithm
│
├── scripts/
│   ├── example_ntp_training.sh   ← Stage 1 training command
│   └── example_rl_training.sh    ← Stage 2 RL training command
│
└── assets/
    └── example_input.mp4         ← Sample video for QUICK_START.md demo
```

---

## Performance Benchmarks

| Metric | Value |
|---|---|
| Token reduction | 4×–100× depending on video content |
| ViT speedup | Up to 19× |
| MLLM speedup | Proportional to visual token reduction² |
| Max resolution supported | 4K (3840×2160) |
| Max frames supported | 1K (1,000 frames) |
| Model size | ~3M parameters |
| Training data | 800K videos (5 sub-datasets) |
| NTP training | 150 epochs, 8× GPU, batch 1024 |
| GRPO training | 1 epoch, 1× GPU, batch 64 |

---

## 3. Findings and Conclusion

AutoGaze represents a pragmatic and principled solution to the video understanding efficiency problem. Its core contributions are: (1) formulating patch selection as an autoregressive RL policy that exploits temporal history, (2) training with a multi-component perceptual reconstruction loss that transfers to classification tasks, (3) enabling any-resolution any-duration processing via a simple tiling strategy, and (4) integrating into existing ViTs via minimal, parameter-free adapter code. The two-stage training pipeline (NTP then GRPO) provides the best of both supervised learning (stable initialization) and reinforcement learning (ability to exceed demonstration quality). The modular codebase architecture (`models/`, `tasks/`, `algorithms/`, `vision_encoders/`) makes it straightforward to extend to new encoders, tasks, and training algorithms.

---

## References

*   Shi, B., Fu, S., Lian, L., Ye, H., Eigen, D., Reite, A., Li, B., Kautz, J., Han, S., Chan, D. M., Molchanov, P., Darrell, T., Yin, H. (2026). *Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing*. CVPR 2026. [arXiv:2603.12254](https://arxiv.org/abs/2603.12254)

*   He, K., Chen, X., Xie, S., Li, Y., Dollar, P., Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022.

*   Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR.

*   Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L. (2023). *Sigmoid Loss for Language Image Pre-Training*. ICCV 2023.

*   Shao, Z., Wang, P., Zhu, Q., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. (Original GRPO paper)

*   Tong, S., Fan, L., Li, Y., et al. (2024). *NVILA: Efficient Frontier Visual Language Models*. (NVILA foundation model)

*   **Repository:** All files at [https://github.com/NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze)
*   **HuggingFace Collection:** [bfshi/autogaze](https://huggingface.co/collections/bfshi/autogaze)
