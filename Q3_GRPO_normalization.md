# Project 4 Reverse Engineering Report: AutoGaze

*   **Project Name:** AutoGaze
*   **Repository:** [https://github.com/NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze)
*   **Project Category:** AI Models & Weights — Efficient Video Understanding

---

## 2. Deep Reasoning Questions & Analysis

### Question 3

> **GRPO (Group Relative Policy Optimization) copies each input group_size times and normalizes advantages relative to the group mean. Why is within-group normalization necessary, and what optimization problem does it solve?**

---

### Direct Answer

Within-group normalization converts absolute reward values (reconstruction loss numbers) into relative advantage signals that are interpretable regardless of current reward scale. Without normalization, if all trajectories for an input are equally poor, the raw reward signals would produce uninformative or destabilizing gradients—either all pushing the policy away equally or all pulling it toward equally bad behaviors. By subtracting the group mean, GRPO ensures that at least some trajectories have positive advantage (better-than-average) and some have negative advantage (worse-than-average), giving the policy a consistent learning signal about *relative* quality, even when absolute quality is low. This is the core function that replaces a separately trained critic/value network.

---

### How to Know the Answer — Repository Evidence

**File:** `TRAIN.md` — GRPO-specific algorithm parameters:

```bash
# From scripts/example_rl_training.sh in TRAIN.md
algorithm.group_size=12 \
algorithm.discount_factor=0.995 \
algorithm.optimize_task_loss_prediction=True
```

**Key parameters explained in `TRAIN.md`:**

```
algorithm.group_size
    (RL only) Number of sampled gazing sequences per input in GRPO. 
    Each input is copied group_size times, and the model samples 
    different gazing sequences. Advantages are computed relative to 
    the group mean. Larger values give more stable training but 
    increase memory.

algorithm.discount_factor
    (RL only) Temporal discount factor for the GRPO advantage. 
    Values close to 1.0 (e.g., 0.995) mean rewards are attributed 
    more evenly across the gazing trajectory; lower values emphasize 
    more recent gazing decisions.
```

**File:** `TRAIN.md` — Gazing ratio configuration for RL:

```bash
model.gazing_ratio_each_frame_config.sample_strategy_during_training=self
```

The `self` strategy means the model first runs with a task-loss constraint to determine per-frame gazing ratios on-policy, then those ratios define how the group_size=12 rollouts distribute patches across frames. The 12 rollouts all start from the same input video, producing 12 different gazing trajectories for comparison.

---

### Proof

**Proof 1 — Group mean substitutes for value function baseline**

Classical policy gradient (REINFORCE) suffers from high variance because it uses raw returns as the learning signal. Actor-critic methods reduce variance by subtracting a learned value function V(s). GRPO replaces V(s) with the empirical mean of the group:

```
Advantage_i = Return_i - mean(Return_1, ..., Return_G)
```

With `group_size=12`, each input generates 12 gazing trajectories. AutoGaze runs VideoMAE on each to get 12 reconstruction loss values (rewards). The mean of these 12 values serves as the baseline. Trajectories with return above the mean get positive advantage (reinforced); below the mean get negative advantage (suppressed). This requires zero additional parameters compared to a learned critic.

**Proof 2 — The discount_factor=0.995 propagates frame-level credit**

AutoGaze selects patches frame-by-frame. A poor patch choice at frame 3 may increase reconstruction loss for frames 3–16 if the context history is degraded. The discount factor `γ=0.995` computes the advantage at gazing step `(t, k)` as:

```
Return(t,k) = -recon_loss(frame_t) × γ^0 
            + -recon_loss(frame_{t+1}) × γ^(patches_remaining_in_frame_t + all_patches_in_frame_{t+1})
            + ...
```

This ensures that the advantage attributed to each individual patch selection reflects its contribution to reconstruction quality across the entire remaining video, not just the current frame.

**Proof 3 — group_size=12 is large enough for stable baseline estimation**

With 12 samples, the variance of the sample mean is σ²/12. In practice, reconstruction loss values for different gazing trajectories on the same video differ by small amounts (similar patches lead to similar reconstruction). A group mean estimated from 12 samples provides a stable enough baseline that gradient noise from single-sample REINFORCE is substantially reduced.

---

### Problem Design

GRPO was originally introduced for LLM reasoning (DeepSeek-R1). AutoGaze adapts it to sequential patch selection by treating each frame's gazing sequence as a "reasoning chain" and the final reconstruction loss as the terminal reward. The within-group normalization is essential because reconstruction loss is measured in absolute units (L1 pixel values) that change as the model improves—early training may have mean loss ≈ 0.8, late training ≈ 0.3. Without normalization, the gradient magnitude would decay as training progresses even though the model continues to improve.

---

### Alternative Suggestions

**PPO (Proximal Policy Optimization)** with a learned value network would provide more accurate advantage estimates but requires training a separate critic network and introduces bootstrapping instability. **REINFORCE with moving average baseline** computes the baseline as an exponential moving average of historical returns—simpler than GRPO but slower to adapt when the distribution of reconstruction losses shifts rapidly. **Direct Policy Optimization (DPO)** would require pairwise preference labels between gazing sequences, which are harder to collect than absolute reconstruction scores.

---

### Summary

Within-group normalization in GRPO solves the high-variance policy gradient problem by providing a locally adaptive baseline (the group mean) computed without any learned critic network. The `group_size=12` configuration in AutoGaze's RL training produces 12 gazing trajectories per video, computes their mean reconstruction reward, and assigns advantages relative to that mean. The `discount_factor=0.995` ensures that the advantage signal correctly attributes credit to individual patch selections across the entire temporal gazing sequence.

---

## 3. Findings and Conclusion

GRPO's within-group normalization is the mechanism that makes RL post-training of AutoGaze stable and parameter-efficient. The `TRAIN.md` parameters `algorithm.group_size=12` and `algorithm.discount_factor=0.995` directly configure this normalization, and the documentation explicitly states that *"advantages are computed relative to the group mean."* This design choice avoids the need for a separate critic network while achieving variance reduction comparable to actor-critic methods.

---

## References

*   Shi, B. et al. (2026). *Attend Before Attention*. CVPR 2026. [arXiv:2603.12254](https://arxiv.org/abs/2603.12254)
*   Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. (Original GRPO paper)
*   Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. (Alternative: PPO)
*   Williams, R. J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. (REINFORCE baseline)
*   **Repository Files:** `TRAIN.md` (`algorithm.group_size`, `algorithm.discount_factor` documentation), `scripts/example_rl_training.sh`
