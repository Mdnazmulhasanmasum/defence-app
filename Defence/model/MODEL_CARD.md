# TinyBrainNet-MT model card
- **Task**: 4-class brain-MRI classification (glioma / meningioma / pituitary / no tumour) + tumour segmentation + class-activation map, one network (unified).
- **Size**: 0.50 M parameters, ONNX 2.02 MB, 16.2 ms on one CPU thread.
- **Data**: BRISC 2025 (CC BY 4.0), official train/test split, image-level split (no patient IDs).
- **Test (ONNX, app pipeline)**: accuracy 97.40%, Dice 0.809.
- **Calibration**: temperature 0.69 (validation). **Triage**: auto >= 0.62, review >= 0.5.
- **Intended use**: research, teaching, decision-support prototype. **NOT** a medical device; not for diagnosis or treatment.
- **Known limits**: single 2-D slices, one dataset family, no radiologist study, no adversarial hardening, no patient-level evaluation.
