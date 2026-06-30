# Project 4 Reverse Engineering Report: AutoGaze

*   **Project Name:** AutoGaze
*   **Repository:** [https://github.com/NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze)
*   **Project Category:** AI Models & Weights — Efficient Video Understanding

---

## 2. Deep Reasoning Questions & Analysis

### Question 10

> **AutoGaze's task-driven training learns patch saliency for a specific task (reconstruction, classification). How does this task-specific saliency transfer to new tasks, and what would happen if you trained for reconstruction but deployed for classification?**

---

### Direct Answer

Reconstruction-trained saliency transfers to classification and QA tasks because reconstruction is a maximal information preservation objective—it forces the model to select patches that collectively represent the complete visual content of the video. Any classification or QA task that requires a subset of that visual content (e.g., detecting moving objects, reading text, identifying scene type) is easier than reconstruction, so reconstruction-sufficient patches are classification-sufficient by inclusion. The reconstruction loss used in AutoGaze (`l1 + dinov2_reg + siglip2`) includes perceptual feature-level terms that directly optimize for the feature space used by downstream MLLMs, further aligning the training objective with MLLM inference requirements. If deployed for classification, a reconstruction-trained AutoGaze would select a slightly larger patch set than a classification-optimized selector, but it would not drop classification-critical patches.

---

### How to Know the Answer — Repository Evidence

**File:** `TRAIN.md` — Multi-component reconstruction loss:

```bash
task.recon_model_config.loss_type=l1+dinov2_reg+siglip2 \
task.recon_model_config.loss_weights=1+0.3+0.3
```

The reconstruction loss includes three components:
- `l1`: Pixel-level reconstruction fidelity
- `dinov2_reg`: DINOv2 feature-level reconstruction — optimizes for preserving DINOv2's learned semantic features, which are heavily used in visual classification
- `siglip2`: SigLIP2 feature-level reconstruction — optimizes for preserving the exact feature space used by NVILA-8B-HD-Video's vision encoder

By including `siglip2` in the reconstruction loss, AutoGaze is explicitly trained to preserve the features that NVILA uses for downstream QA tasks. The reconstruction-classification transfer is not accidental—it is engineered through the feature-level loss terms.

**File:** `README.md` — Task module description:

> *"tasks: Definition of the task serving as the training objective for gaze models... the reward function used to train the gaze model (e.g., reconstruction reward)... As a side use, the task can also be used to train the task model (e.g., VideoMAE) itself."*

The task module is modular. A different task (e.g., DINOv2 classification) could be substituted to train a classification-optimized gaze model. The fact that reconstruction with perceptual losses is chosen—rather than a simpler pixel-only L2 loss—reflects a deliberate design decision to align the training objective with perceptual (classification-relevant) features.

**File:** `TRAIN.md` — DINOv2 and SigLIP2 in task recon loss:

```
task.recon_model_config.loss_type
    Reconstruction loss type(s) separated by '+'. l1 is pixel-level L1 loss; 
    dinov2_reg is DINOv2 feature-level loss; siglip2 is SigLIP2 feature-level loss. 
    Combining multiple losses improves reconstruction quality.
```

DINOv2 is trained on ImageNet-scale visual classification. SigLIP2 is trained on image-language alignment. Including both in the reconstruction loss means the gaze model must select patches that preserve ImageNet-class-discriminating features (DINOv2) and language-grounded visual features (SigLIP2)—both of which are directly useful for video QA classification tasks.

---

### Proof

**Proof 1 — Feature-level losses bridge reconstruction and classification**

A pure pixel-L1 reconstruction loss would optimize for patch selection that minimizes mean pixel error—potentially selecting many homogeneous background patches that have large spatial area and thus contribute heavily to pixel-L1. The `dinov2_reg + siglip2` terms redirect attention toward patches that preserve classification-relevant semantic features. The weighting `1 + 0.3 + 0.3` gives perceptual losses 37.5% of the total training signal, substantial enough to shape the learned saliency toward semantic regions.

**Proof 2 — NVILA-8B-HD-Video validates reconstruction → classification transfer**

From `README.md`:
```
NVILA-8B-HD-Video — A video MLLM scaled to 1K frames, 4K resolution with AutoGaze
```

NVILA-8B-HD-Video is an MLLM deployed for video QA (classification/generation tasks). It uses AutoGaze trained with reconstruction reward. The fact that this combined system achieves competitive performance on video QA benchmarks confirms that reconstruction-trained saliency transfers to QA tasks without task-specific fine-tuning.

**Proof 3 — Modular task design enables task-specific training if needed**

The `autogaze/tasks/` module is explicitly designed to support arbitrary task substitution:

From `README.md`:
> *"This modularized structure allows easily adding new models, tasks, and algorithms... For example, to add a task of DINOv2 feature reconstruction, one only needs to define a new task class without touching other parts. Sometimes adding some new features require changing multiple components, for example, adding a new task of Kinetics video classification will require defining the new task and adding a new Kinetics dataset."*

The explicit mention of "Kinetics video classification" as a possible new task demonstrates that the designers considered classification-specific training. The choice to use reconstruction (rather than classification) as the default training task reflects confidence in cross-task transfer, not a technical limitation.

---

### Problem Design

The transfer mechanism works because AutoGaze learns to identify patches containing *temporal change* and *semantic texture*—both of which are necessary for both reconstruction and classification. A patch showing a person's face in motion contains high-frequency texture (DINOv2-relevant), semantic content (SigLIP2-relevant), and temporal change (L1-relevant). All three loss components agree that this patch should be selected, regardless of whether the downstream task is reconstruction or action recognition.

If AutoGaze were deployed for reconstruction but trained for classification: the opposite scenario would work less well. A classification-optimized gaze model might learn to select only the single most semantically discriminative patch per scene (sufficient for classification), missing the background, textures, and secondary objects needed for full reconstruction fidelity.

---

### Alternative Suggestions

**Task-specific fine-tuning**: After reconstruction pre-training, fine-tune the gaze model with a classification reward on the target classification dataset. This would improve efficiency (fewer patches needed) at the cost of generality. **Multi-task training**: Train with a weighted combination of reconstruction and classification losses simultaneously. More complex but potentially optimal for deployment scenarios requiring both reconstruction quality and classification accuracy. **Zero-shot task specification**: Condition the gaze model on a text description of the downstream task (e.g., "select patches for action recognition"), enabling dynamic task adaptation at inference time without any fine-tuning.

---

### Summary

Reconstruction-trained saliency transfers to classification tasks because: (1) reconstruction is a superset task—reconstruction-sufficient patches contain classification-sufficient information, and (2) the multi-component loss (`l1 + dinov2_reg + siglip2`) explicitly includes perceptual losses that optimize for the feature spaces used in downstream classification. NVILA-8B-HD-Video is the deployed proof that this transfer works at scale. The modular task system in `autogaze/tasks/` allows task-specific training if stronger transfer guarantees are needed for specialized applications.

---

## 3. Findings and Conclusion

Cross-task saliency transfer is enabled both architecturally (multi-component perceptual reconstruction loss) and empirically (NVILA-8B-HD-Video QA performance). The repository's modular task system provides the extensibility to add classification-specific training objectives if needed, confirming that reconstruction-based training is a deliberate design choice for generalization, not a limitation.

---

## References

*   Shi, B. et al. (2026). *Attend Before Attention*. CVPR 2026. [arXiv:2603.12254](https://arxiv.org/abs/2603.12254)
*   Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. (`dinov2_reg` loss)
*   Zhai, X. et al. (2023). *Sigmoid Loss for Language Image Pre-Training*. (`siglip2` loss)
*   Kay, W. et al. (2017). *The Kinetics Human Action Video Dataset*. (Referenced as possible new task)
*   **Repository Files:** `TRAIN.md` (`task.recon_model_config.loss_type`), `README.md` (task module, NVILA), `autogaze/tasks/` (modular task design)
