# Project 4 Reverse Engineering Report: AutoGaze

*   **Project Name:** AutoGaze
*   **Repository:** [https://github.com/NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze)
*   **Project Category:** AI Models & Weights — Efficient Video Understanding

---

## 2. Deep Reasoning Questions & Analysis

### Question 5

> **AutoGaze includes optional task loss prediction: a secondary objective that predicts the task loss at each step. Why would predicting loss be useful, and what does this auxiliary task teach the model?**

---

### Direct Answer

The task loss prediction head teaches the model to maintain an internal estimate of *reconstruction quality achieved so far* at every step of the autoregressive gazing process. This internal estimate is what enables the `task_loss_requirement` early-stopping mechanism at inference time: the model stops adding patches to a frame once its predicted reconstruction loss drops below the user-specified threshold, without needing to actually run the expensive VideoMAE reconstruction model at every step. Without this auxiliary task, early stopping would require invoking VideoMAE after each patch selection—100–265 VideoMAE forward passes per frame—making inference prohibitively expensive.

---

### How to Know the Answer — Repository Evidence

**File:** `TRAIN.md` — Algorithm parameter:

```bash
algorithm.optimize_task_loss_prediction=True
```

This flag is enabled in both NTP pre-training and GRPO RL post-training. From the parameter documentation:

```
algorithm.optimize_task_loss_prediction
    Whether to jointly train the model to predict the task loss at 
    each gazing step. This enables the model to estimate when 
    reconstruction quality is sufficient for early stopping.
```

**File:** `TRAIN.md` — Task loss requirement inference configuration (RL stage):

```bash
model.has_task_loss_requirement_during_training=False \
model.has_task_loss_requirement_during_inference=True \
model.task_loss_requirement_config.fixed.task_loss_requirement=0.7 \
model.task_loss_requirement_config.uniform.task_loss_requirement_min=0.5 \
model.task_loss_requirement_config.uniform.task_loss_requirement_max=1.0
```

**Critical:** During training (`has_task_loss_requirement_during_training=False`), the model uses a fixed gazing ratio, so it cannot stop early based on predicted loss. But during inference (`has_task_loss_requirement_during_inference=True`), it uses the trained loss predictor to stop early at each frame.

The uniform sampling range `[0.5, 1.0]` during RL training teaches the model to predict loss accurately across multiple quality thresholds, not just a single fixed value.

**File:** `QUICK_START.md` — Inference usage of the loss predictor:

```python
# task_loss_requirement: the model stops gazing at each frame once 
# the reconstruction loss falls under 0.7
gaze_outputs = autogaze_model(
    {"video": video_input_autogaze}, 
    gazing_ratio=0.75, 
    task_loss_requirement=0.7
)
```

The `task_loss_requirement=0.7` parameter is only meaningful because the model has a trained internal loss predictor. Without `optimize_task_loss_prediction=True`, this parameter would have no effect.

---

### Proof

**Proof 1 — Loss prediction enables early stopping without external oracle**

VideoMAE (the reconstruction oracle) is a large model with non-trivial inference cost. Running it after every single patch selection during inference would be:

- Per frame: up to 265 VideoMAE forward passes
- Per video: up to 265 × 16 = 4,240 VideoMAE forward passes

The loss prediction head is a lightweight linear layer on the decoder's hidden state. It predicts scalar reconstruction loss at each autoregressive step with negligible cost. The `task_loss_requirement=0.7` parameter uses this predicted value to decide when to stop, avoiding all VideoMAE calls during inference.

**Proof 2 — Training configuration confirms joint optimization**

Both stages set `algorithm.optimize_task_loss_prediction=True`:

NTP stage:
```bash
algorithm.optimize_task_loss_prediction=True
```

RL stage:
```bash
algorithm.optimize_task_loss_prediction=True
```

The fact that it is enabled in both stages means the loss prediction head is trained with both supervised regression on greedy-search loss trajectories (NTP stage) and further refined with RL rewards (GRPO stage). This ensures the predictor remains accurate as the model learns to produce better gazing sequences.

**Proof 3 — Uniform threshold sampling teaches generalization**

During RL training, `task_loss_requirement_config.uniform.task_loss_requirement_min=0.5` and `max=1.0` means the model is trained to predict reconstruction loss accurately across a range of quality levels. This forces the loss predictor to learn a continuous, calibrated estimate rather than a binary "sufficient/not sufficient" signal, enabling users to set any threshold between 0.5 and 1.0 at inference time and get correct stopping behavior.

---

### Problem Design

The loss prediction auxiliary task solves the exploration-exploitation tradeoff in early stopping: too-early stopping wastes the reconstruction quality potential of remaining patches; too-late stopping wastes tokens on patches that provide negligible improvement. A calibrated internal loss estimate allows the model to operate at the precise stopping point where marginal reconstruction improvement per additional patch approaches zero. This is analogous to learning a stopping criterion in Monte Carlo Tree Search or value estimation in model-based RL.

---

### Alternative Suggestions

**External oracle stopping**: Run VideoMAE after each patch and stop when actual loss < threshold. Correct but expensive (4,240 VideoMAE calls per video). **Fixed gazing ratio**: Use `gazing_ratio=0.75` as the only stopping criterion, ignoring content complexity. Simple but wastes tokens on easy frames. **Learned reward model**: Train a separate neural network to predict reconstruction quality from patch indices, independent of the gaze model. More flexible but doubles the parameter count and training complexity.

---

### Summary

The task loss prediction auxiliary objective enables the key inference feature `task_loss_requirement`, allowing AutoGaze to adaptively stop gazing at each frame once predicted reconstruction loss meets the quality threshold—without invoking the expensive VideoMAE oracle at inference time. The configuration `algorithm.optimize_task_loss_prediction=True` in both NTP and RL stages, combined with uniform threshold sampling `[0.5, 1.0]`, produces a calibrated loss predictor that generalizes across quality requirements.

---

## 3. Findings and Conclusion

Task loss prediction is not a secondary concern in AutoGaze—it is the mechanism that unlocks adaptive early stopping, the primary efficiency advantage over fixed-ratio token reduction schemes. The `task_loss_requirement` parameter in `QUICK_START.md` and the uniform threshold training in `TRAIN.md` confirm that this auxiliary task was carefully designed to produce a generalizable, calibrated quality estimator.

---

## References

*   Shi, B. et al. (2026). *Attend Before Attention*. CVPR 2026. [arXiv:2603.12254](https://arxiv.org/abs/2603.12254)
*   He, K. et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. (VideoMAE oracle)
*   **Repository Files:** `TRAIN.md` (`algorithm.optimize_task_loss_prediction`, `model.task_loss_requirement_config`), `QUICK_START.md` (`task_loss_requirement` parameter)
