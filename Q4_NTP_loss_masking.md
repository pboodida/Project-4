# Project 4 Reverse Engineering Report: AutoGaze

*   **Project Name:** AutoGaze
*   **Repository:** [https://github.com/NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze)
*   **Project Category:** AI Models & Weights — Efficient Video Understanding

---

## 2. Deep Reasoning Questions & Analysis

### Question 4

> **NTP (Next Token Prediction) uses loss masking to ignore padded positions. Why is masking necessary, and what training failure would occur without it?**

---

### Direct Answer

Loss masking is necessary because AutoGaze generates variable-length gazing sequences per frame—early frames select many patches, later frames select far fewer due to temporal redundancy. When multiple samples are batched together, shorter sequences must be padded to the length of the longest sequence. Padded positions do not correspond to real patch selections; they are dummy tokens inserted for alignment. Computing cross-entropy loss at padded positions would force the model to predict meaningless dummy tokens, corrupting the learned probability distribution over real patch indices and producing degenerate inference behavior where the model allocates probability mass to invalid positions.

---

### How to Know the Answer — Repository Evidence

**File:** `QUICK_START.md` — Padded gazing in inference output:

```python
print(gaze_outputs['gazing_pos'].shape)
# 1 * 348
# gazing_pos records the indices of the patches being gazed at.
# This means AutoGaze gazed at 348 patches (including padded gazing) for the video.

print(gaze_outputs['if_padded_gazing'].shape)
# 1 * 348
# if_padded_gazing records which positions are padded (dummy) gazing.
# Each element is boolean, True means the gazing at that position is padded.

print((~gaze_outputs['if_padded_gazing']).sum(dim=-1))
# 213 — actual number of real gazed patches

print(gaze_outputs['num_gazing_each_frame'])
# [198, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
```

**Critical observation:** 348 total positions, 213 real, 135 padded. The `if_padded_gazing` boolean mask precisely identifies which positions must be excluded from NTP loss computation. Frame 1 has 198 patches; frames 2–16 each have 10 patches. If all 16 frames were padded to 198, approximately (198×16 - 213) / (198×16) ≈ 93.3% of positions would be padding—overwhelmingly noise if included in the loss.

**File:** `INTEGRATION.md` — SigLIP adapter padding behavior:

```python
def mask_with_gazing(self, sequence, gazing_info):
    """
    Select only the gazed patches from the full sequence.
    Padded gazing positions are mapped to a dummy token (index 0).
    """
    gazing_pos = gazing_info['gazing_pos'].clone()
    if_padded_gazing = gazing_info['if_padded_gazing'].clone()

    B = sequence.shape[0]

    # Map padded gazing positions to a dummy token
    gazing_pos[if_padded_gazing] = 0

    # Gather only the gazed tokens
    sequence_gazed = sequence[torch.arange(B)[:, None], gazing_pos]
    return sequence_gazed
```

This shows that at inference time, padded positions are assigned dummy index 0. During training, the NTP loss mask prevents the model from ever receiving gradient updates at those positions.

**File:** `QUICK_START.md` — Post-processing note:

```python
# Only keep the features at non-padded gazing positions
last_hidden_state = [
    f[~if_pad] for f, if_pad in 
    zip(siglip_outputs.last_hidden_state, gaze_outputs['if_padded_gazing'])
]
```

This confirms that padded positions are systematically excluded at every stage of the pipeline—training (via loss mask), inference (via `if_padded_gazing`), and downstream usage (via post-processing filter).

---

### Proof

**Proof 1 — Variable sequence lengths require padding**

From the output `[198, 10, 10, ..., 10]`, frame 1 selects 198 patches while frames 2–16 each select 10. In a batch of multiple videos, sequence lengths will vary further. PyTorch's DataLoader requires fixed-shape tensors. The autoregressive decoder must output a fixed-length token sequence per batch item, so the actual gazing sequence (length 213) is padded to batch-max length (348 in this example). Without a mask, the cross-entropy loss would be computed at all 348 positions, with 135 positions providing gradient signal toward predicting the dummy token instead of valid patch indices.

**Proof 2 — Without masking: probability leak to invalid tokens**

The vocabulary size is `num_vision_tokens_each_frame=265` distinct patch indices per frame. If the dummy padding token (e.g., a sentinel index) receives gradient signals from 135/348 ≈ 38.8% of training positions, the softmax output would learn to assign non-trivial probability to that dummy index even during inference when no padding exists. This creates a probability leak: at each autoregressive decoding step, a fraction of probability mass is wasted on an unattainable dummy token, reducing the effective probability mass over valid 265-token vocabulary. The model would systematically under-select real patches.

**Proof 3 — The mask is explicitly tracked through the entire pipeline**

The `if_padded_gazing` key is returned as a first-class output from `autogaze_model()` and is passed into `siglip_model(gazing_info=gaze_outputs)`. The SigLIP adapter uses it in `get_causal_mask()` to zero out attention from padded positions:

```python
# From INTEGRATION.md
# Zero out columns for padded tokens
if token_mask is not None:
    mask = mask * (~token_mask.unsqueeze(1)).to(dtype)
```

This pervasive use of the padding mask confirms that its existence is architecturally fundamental, not an optional cleanup step.

---

### Problem Design

The variable-length gazing sequence arises from AutoGaze's design goal: select the *minimum* number of patches needed for reconstruction. First frames require many patches (no context history), while later frames require few (static regions already covered). This content-adaptive behavior is the core efficiency mechanism, but it creates variable-length sequences that must be padded for batched training. The `if_padded_gazing` mask is the bridge between the variable-length semantic sequence and the fixed-length computational representation.

---

### Alternative Suggestions

**Dynamic batching** (grouping sequences of the same length) would eliminate padding but dramatically complicates the DataLoader and reduces GPU utilization due to smaller effective batch sizes. **Packing** (concatenating multiple short sequences into one fixed-length buffer with segment IDs) is used in LLM training and could work here—it eliminates padding waste but requires rewriting the attention mask to prevent cross-sequence attention. Given AutoGaze's block-causal attention mask (already complex), packing would further complicate the masking logic.

---

### Summary

NTP loss masking is essential because AutoGaze produces variable-length gazing sequences (198 patches for frame 1, 10 for frames 2–16) that must be padded to enable batched training. Without masking, 38–93% of gradient updates per batch would push the model toward predicting dummy padding tokens, corrupting the learned patch-selection distribution. The `if_padded_gazing` boolean tensor is the implementation artifact that enforces this masking throughout training, inference, and downstream processing.

---

## 3. Findings and Conclusion

Loss masking in NTP training is not an implementation detail—it is a fundamental requirement imposed by the variable-length nature of AutoGaze's adaptive gazing sequences. The `if_padded_gazing` field visible in `QUICK_START.md`'s inference output demonstrates the pervasiveness of this concern across the entire pipeline, from training loss computation through SigLIP attention masking to final feature extraction.

---

## References

*   Shi, B. et al. (2026). *Attend Before Attention*. CVPR 2026. [arXiv:2603.12254](https://arxiv.org/abs/2603.12254)
*   Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. (Padding mask in transformer training)
*   **Repository Files:** `QUICK_START.md` (`if_padded_gazing`, `num_gazing_each_frame` outputs), `INTEGRATION.md` (`mask_with_gazing`, `get_causal_mask`), `TRAIN.md`
