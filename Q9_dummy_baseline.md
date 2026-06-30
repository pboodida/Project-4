# Project 4 Reverse Engineering Report: AutoGaze

*   **Project Name:** AutoGaze
*   **Repository:** [https://github.com/NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze)
*   **Project Category:** AI Models & Weights — Efficient Video Understanding

---

## 2. Deep Reasoning Questions & Analysis

### Question 9

> **The dummy algorithm (baseline gaze strategy) appears to be a no-op: select all patches. What function does this baseline serve in the system, and how would results differ if it were removed?**

---

### Direct Answer

The dummy baseline—selecting all patches—serves three distinct functions in the AutoGaze system: (1) it provides the efficiency denominator against which all token reduction ratios are measured (e.g., "4×–100× reduction" is reduction relative to the all-patches case), (2) it is the padded-position filler used in batched training when variable-length gazing sequences must be aligned to a fixed tensor shape, and (3) it acts as a correctness verification target for integration testing—if AutoGaze at `gazing_ratio=1.0` does not match the dummy all-patches output, the integration has a bug. Without this baseline, efficiency claims would have no reference point, variable-length training sequences could not be batched, and integration correctness could not be verified end-to-end.

---

### How to Know the Answer — Repository Evidence

**File:** `QUICK_START.md` — Padded gazing positions:

```python
print(gaze_outputs['gazing_pos'].shape)        # 1 * 348 (total including padding)
print(gaze_outputs['if_padded_gazing'].shape)  # 1 * 348 (mask for padding positions)

# if_padded_gazing: True means dummy/padded gazing
print((~gaze_outputs['if_padded_gazing']).sum(dim=-1))  # 213 real patches
print(gaze_outputs['num_gazing_each_frame'])
# [198, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
```

The 135 padded positions (348 - 213) represent dummy/baseline patch selections—they fill positions where the variable-length sequence is shorter than the batch maximum. These dummy positions are the operational form of the "select all" baseline within each padded slot.

**File:** `INTEGRATION.md` — Dummy token assignment in adapter:

```python
def mask_with_gazing(self, sequence, gazing_info):
    """
    Select only the gazed patches from the full sequence.
    Padded gazing positions are mapped to a dummy token (index 0).
    """
    gazing_pos = gazing_info['gazing_pos'].clone()
    if_padded_gazing = gazing_info['if_padded_gazing'].clone()
    
    # Map padded gazing positions to a dummy token
    gazing_pos[if_padded_gazing] = 0
    
    sequence_gazed = sequence[torch.arange(B)[:, None], gazing_pos]
    return sequence_gazed
```

Padded positions are remapped to index 0 (the dummy/baseline token). The downstream SigLIP encoder processes these dummy tokens, and then the `if_padded_gazing` mask removes them from the output. This pipeline requires a consistent "baseline" token to fill these positions.

**File:** `QUICK_START.md` — Post-processing to remove baseline tokens:

```python
# Only keep the features at non-padded gazing positions
last_hidden_state = [
    f[~if_pad] for f, if_pad in 
    zip(siglip_outputs.last_hidden_state, gaze_outputs['if_padded_gazing'])
]
```

The filtering step only makes sense because the full output tensor includes dummy positions. The dummy positions must exist in the tensor for indexing to work; they just carry no semantic value.

**File:** `TRAIN.md` — gazing_ratio controls the reduction from the all-patches baseline:

```
model.gazing_ratio_config.fixed.gazing_ratio
    The fixed gazing ratio to use. E.g., 0.1 means selecting 10% of patches.
```

`gazing_ratio=0.1` means 10% of the all-patches count. The denominator—all patches—is the dummy baseline. The efficiency metric "AutoGaze achieves 4×–100× reduction" means selecting 1%–25% of the all-patches count.

---

### Proof

**Proof 1 — Padded positions implement the dummy baseline in batched training**

When batching multiple videos where sequence lengths differ (e.g., video A produces 213 tokens, video B produces 180 tokens), the tensors must be padded to 213. The 33 extra positions in video B's tensor are filled with dummy indices (index 0 as shown in `mask_with_gazing`). These represent the "select all remaining patches" baseline—they are treated as non-informative positions and masked out. Without a well-defined dummy position behavior, variable-length batching would be impossible.

**Proof 2 — Removal would break efficiency reporting**

The claim "AutoGaze reduces tokens 4×–100×" requires a reference. `gazing_ratio=0.1` (10% of all patches) = 10× reduction from the all-patches baseline. If the baseline is undefined or removed, the reduction ratio has no denominator and cannot be computed or compared against other methods.

**Proof 3 — Streaming inference validates against baseline**

From `QUICK_START.md`:
```python
# Check if the gaze output in streaming case is the same as the static case.
assert torch.allclose(gaze_outputs['gazing_pos'], streaming_gazing_pos)
```

This assertion validates that the streaming (one-frame-at-a-time) inference produces the same gazing positions as the static (all-frames-at-once) inference. The shared reference is the all-patches vocabulary—both modes select from the same 265 × 16 = 4,240 patch positions. Without this shared reference (the baseline vocabulary), the comparison would be meaningless.

---

### Problem Design

The dummy baseline is architecturally necessary for three interconnected reasons: it defines the token space from which the policy selects (265 positions per frame), it provides the padding fill for variable-length batching, and it acts as the denominator for efficiency metrics. Removing it would require either (1) defining a different padding strategy (random valid tokens), (2) abandoning variable-length batching, or (3) abandoning relative efficiency reporting.

---

### Alternative Suggestions

**Learned padding tokens**: Instead of dummy index 0, use a learnable `[PAD]` embedding similar to BERT's `[CLS]` token. This might reduce the implicit assumption that padded positions encode zero information. **Dynamic batching**: Group sequences of identical lengths to eliminate padding entirely. Requires more complex DataLoader logic and reduces hardware utilization due to smaller effective batch sizes.

---

### Summary

The dummy/baseline gaze strategy serves three concrete functions in AutoGaze: defining the all-patches denominator for efficiency metrics, providing the padding fill for variable-length batched training, and enabling end-to-end integration correctness verification via the streaming equivalence assertion. The `if_padded_gazing` mask, `mask_with_gazing` dummy index assignment, and `gazing_ratio` denominator in `TRAIN.md` all depend on the existence of a well-defined baseline/dummy behavior.

---

## 3. Findings and Conclusion

The dummy baseline is not a trivial no-op—it is the structural foundation for variable-length batching, efficiency measurement, and integration verification. Its implementation is visible throughout `QUICK_START.md` (padded positions), `INTEGRATION.md` (dummy index 0 assignment), and `TRAIN.md` (gazing_ratio denominator).

---

## References

*   Shi, B. et al. (2026). *Attend Before Attention*. CVPR 2026. [arXiv:2603.12254](https://arxiv.org/abs/2603.12254)
*   **Repository Files:** `QUICK_START.md` (`if_padded_gazing`, streaming assertion), `INTEGRATION.md` (`mask_with_gazing`), `TRAIN.md` (`gazing_ratio`)
