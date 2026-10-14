# TEST REPORT: Explainable Skin Cancer Detection System

**Date**: October 14, 2026  
**Environment**: Windows 11 / Python 3.12.4 (.venv)  
**Status**: ALL TESTS PASSED (100% VERIFIED)

---

## 1. Summary of Test Execution

| Test ID | Test Name | Target Component | Method | Verification Result | Status |
|---|---|---|---|---|---|
| **T1** | Canonical 7-Class Definition | `dataset.py` | Unit Test | Exact match `['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']` | **PASS** |
| **T2** | Bidirectional Index Mapping | `dataset.py` | Unit Test | Strict 1-to-1 mapping for all 7 classes verified | **PASS** |
| **T3** | Metadata Schema Validation | `dataset.py` | Unit Test | Validates `image_id`, `dx`, rejects unknown classes | **PASS** |
| **T4** | Lesion-Group Leakage Prevention | `dataset.py` / `train_model.py` | Unit Test | Disjoint `lesion_id` split verified (0% leakage) | **PASS** |
| **T5** | Demo Sample Image Generator | `dataset.py` | Unit Test | Generated 224x224 RGB test lesions | **PASS** |
| **T6** | Model Instantiation & Top Head | `train_model.py` | Unit / Smoke | MobileNetV2 + 7-unit Softmax classification head | **PASS** |
| **T7** | Full Model Training & Checkpointing | `train_model.py` | Local Training | Trained on 6,959 images, saved best weights (`val_loss: 1.2412`) | **PASS** |
| **T8** | Test-Set Evaluation Script | `evaluate_model.py` | Batch Evaluation | Evaluated 1,527 test images (59.40% Acc, 0.6215 Weighted F1) | **PASS** |
| **T9** | Grad-CAM Gradient & Overlay | `gradcam.py` | Verification Script | `out_relu` activation gradients, colormap, 224x224 overlay | **PASS** |
| **T10** | Streamlit UI & Visual Dashboard | `app.py` | Browser Subagent | UI rendered, predictions displayed, Grad-CAM & eval verified | **PASS** |

---

## 2. Test Execution Details

### T1–T5: Pipeline Unit Tests
```text
Ran 5 tests in 4.503s
OK
```

### T6–T7: Transfer Learning Training
- **Dataset**: HAM10000 (10,015 images matched across `HAM10000_images_part_1` and `HAM10000_images_part_2`)
- **Partitions (Leakage-Free via `lesion_id`)**:
  - Train: 6,959 images
  - Validation: 1,529 images
  - Test: 1,527 images
- **Class Weights**: Balanced inverse frequency applied
- **Training Duration**: 3 epochs (~12 minutes on CPU)
- **Validation Loss**: Improved from 1.3496 to 1.2412 (Best checkpoint saved)
- **Artifacts Saved**:
  - `models/skin_lesion_model.keras`
  - `models/classes.json`
  - `models/model_config.json`
  - `results/training_curves.png`
  - `results/training_history.json`

### T8: Held-Out Test Set Evaluation
- **Total Test Images**: 1,527
- **Test Accuracy**: **59.40%**
- **Weighted F1-Score**: **0.6215**
- **Macro F1-Score**: **0.3088**
- **Per-Class Support**:
  - `nv`: 1,023 images
  - `mel`: 170 images
  - `bkl`: 168 images
  - `bcc`: 78 images
  - `akiec`: 50 images
  - `vasc`: 21 images
  - `df`: 17 images
- **Saved Evaluation Files**:
  - `results/evaluation_report.json` & `models/evaluation_report.json`
  - `results/confusion_matrix.png` & `models/confusion_matrix.png`

### T9: Grad-CAM Explainability Test
```text
GRADCAM_VERIFIED_SUCCESSFULLY:
- Target Layer: out_relu
- Raw Heatmap Shape: (7, 7) [Values in 0.0 to 1.0]
- Blended Overlay Shape: (224, 224, 3) [uint8 RGB]
```

### T10: Streamlit Live Browser Verification
- **Local URL**: `http://localhost:8501`
- **Sidebar Status**: `Model Status: Available (Loaded)`
- **Inference & Grad-CAM**: Verified interactive prediction and 3-column Grad-CAM overlay display
- **Evaluation Dashboard**: Verified test metrics and confusion matrix image display
