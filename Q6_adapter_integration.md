# Project 4 Reverse Engineering Report: AutoGaze

*   **Project Name:** AutoGaze
*   **Repository:** [https://github.com/NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze)
*   **Project Category:** AI Models & Weights — Efficient Video Understanding

---

## 2. Deep Reasoning Questions & Analysis

### Question 6

> **AutoGaze integrates with existing vision models (SigLIP, MLLMs) via adapters rather than modifying these models. Why is adapter-based integration superior to end-to-end fine-tuning?**

---

### Direct Answer

Adapter-based integration preserves the pre-trained feature representations of the vision encoder, avoids catastrophic forgetting of the large-scale image-language pre-training, and makes AutoGaze encoder-agnostic—the same gaze model can be attached to any ViT with minimal code changes. End-to-end fine-tuning would co-optimize the vision encoder's internal feature representations for the patch selection task, degrading its general-purpose visual understanding and eliminating its reusability across different downstream tasks. The adapter approach requires modifying only two components: the patch embedding layer (to select only gazed patches) and the attention mask (to control inter-frame interactions), while leaving all encoder layers, MLP blocks, and layer norms completely unchanged.

---

### How to Know the Answer — Repository Evidence

**File:** `INTEGRATION.md` — Summary of changes table:

```
| Component              | Original ViT          | With AutoGaze                          |
|------------------------|-----------------------|----------------------------------------|
| Input shape            | (B, C, H, W)          | (B, T, C, H, W)                        |
| Patch embedding        | Embeds all patches    | Embeds only gazed patches via mask_with_gazing |
| Attention mask         | None (full attention) | Block-causal / causal / bidirectional from get_causal_mask |
| Encoder / MLP / LayerNorm | No change          | No change                              |
| Config                 | Standard              | + scales, attn_type, frame_independent_encoding |
```

**Encoder / MLP / LayerNorm: No change** — this is the definitive statement that confirms adapter-only integration. The 400M+ parameters of SigLIP-SO400M are not modified.

**File:** `INTEGRATION.md` — Patch embedding modification:

```python
def mask_with_gazing(self, sequence, gazing_info):
    """
    Select only the gazed patches from the full sequence.
    Padded gazing positions are mapped to a dummy token (index 0).
    """
    gazing_pos = gazing_info['gazing_pos'].clone()
    if_padded_gazing = gazing_info['if_padded_gazing'].clone()
    B = sequence.shape[0]
    gazing_pos[if_padded_gazing] = 0
    sequence_gazed = sequence[torch.arange(B)[:, None], gazing_pos]
    return sequence_gazed
```

This is a pure indexing operation—no learnable parameters added to the patch embedding process. The original patch embedding weights are used unchanged; the adapter simply selects which output tokens to forward.

**File:** `INTEGRATION.md` — Attention mask modification:

```python
def get_causal_mask(self, num_tokens_each_frame, batch_size, num_heads,
                    token_mask=None, dtype=torch.float32):
    T = len(num_tokens_each_frame)
    N = num_tokens_each_frame.sum()

    # Start with a causal (lower-triangular) mask
    mask = torch.tril(torch.ones(batch_size, N, N, dtype=dtype))

    # Allow full bidirectional attention within each frame
    for t in range(T):
        start = num_tokens_each_frame[:t].sum()
        end = num_tokens_each_frame[:t+1].sum()
        mask[:, start:end, start:end] = 1

    # Zero out columns for padded tokens
    if token_mask is not None:
        mask = mask * (~token_mask.unsqueeze(1)).to(dtype)

    mask = torch.where(mask == 1, 0, -torch.inf).to(dtype)
    mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
    return mask
```

Again, no new learnable parameters—only a masking function applied to the existing attention computation.

**File:** `TRAIN.md` — Frozen task model:

```bash
trainer.train_task=False \
trainer.detach_task=True
```

The task model (VideoMAE) is frozen during AutoGaze training. SigLIP, when used as a downstream encoder, is similarly kept frozen.

---

### Proof

**Proof 1 — Zero new parameters in the adapter**

`mask_with_gazing` is a parameter-free indexing operation. `get_causal_mask` is a parameter-free masking function. The config additions (`scales`, `attn_type`, `frame_independent_encoding`) are string/bool fields that control behavior without adding weights. The AutoGaze gaze model itself (3M parameters) is the only new trainable component in the entire pipeline.

**Proof 2 — Multiple encoder compatibility**

From `INTEGRATION.md`:
> *"This guide explains how to modify an existing image ViT to work with AutoGaze."*

The guide is written as a general recipe, not a SigLIP-specific modification. The repository references both `SigLIP` and `DINOv2` as compatible encoders:

From `README.md`:
> *"vision_encoders: Vision encoders that can be used with AutoGaze. Here you can customize existing vision encoders such as SigLIP or DINOv2 to make them compatible with AutoGaze."*

End-to-end fine-tuning would produce a SigLIP variant that is incompatible with standard SigLIP checkpoints, requiring re-training for each new encoder.

**Proof 3 — Attention mask backend compatibility is explicit**

`INTEGRATION.md` documents three `attn_type` options (`block_causal`, `causal`, `bidirectional`) and their backend compatibility with `flash_attention_2`, `sdpa`, `eager`, and `flex_attention`. This level of compatibility planning only makes sense if the encoder's internal attention implementation remains unchanged—the adapter merely injects a masking tensor that the existing attention mechanism interprets.

---

### Problem Design

The adapter approach makes AutoGaze a drop-in module: any team with an existing ViT-based video pipeline can add AutoGaze by implementing `mask_with_gazing` and `get_causal_mask` in their embedding and transformer modules. If AutoGaze required end-to-end fine-tuning, each deployment would need re-training the full ViT, creating a new proprietary model variant that loses compatibility with the original pre-trained weights and the broader ecosystem of fine-tuned variants.

---

### Alternative Suggestions

**LoRA-based integration** would allow small learned adapter matrices inside the ViT's attention layers rather than zero-parameter masking. This could potentially improve feature adaptation to sparse multi-scale input but adds parameters and training complexity. **Prompt tuning** (learned visual prompts prepended to the token sequence) is another parameter-efficient approach but does not address the fundamental need to change which patches are input to the encoder.

---

### Summary

Adapter-based integration preserves the full pre-trained representation capacity of the vision encoder while adding zero learnable parameters to it. The two required modifications—`mask_with_gazing` (patch selection filter) and `get_causal_mask` (inter-frame attention control)—are both parameter-free and require no gradient computation in the encoder. `INTEGRATION.md` explicitly confirms that `Encoder / MLP / LayerNorm: No change`, making AutoGaze fully compatible with any unmodified SigLIP or DINOv2 checkpoint.

---

## 3. Findings and Conclusion

The adapter design choice is validated by the zero-parameter modification to the vision encoder and the explicit multi-encoder compatibility (SigLIP, DINOv2) described in `README.md` and `INTEGRATION.md`. This makes AutoGaze a practical, deployable solution rather than a research prototype requiring full pipeline re-training for each deployment target.

---

## References

*   Shi, B. et al. (2026). *Attend Before Attention*. CVPR 2026. [arXiv:2603.12254](https://arxiv.org/abs/2603.12254)
*   Zhai, X. et al. (2022). *Scaling Vision Transformers*. (SigLIP backbone)
*   Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. (Alternative adapter approach)
*   **Repository Files:** `INTEGRATION.md` (full integration guide), `README.md` (vision encoder support), `TRAIN.md` (`trainer.train_task=False`)
