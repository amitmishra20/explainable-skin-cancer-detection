# Explainable Skin Cancer Detection Using CNN (MobileNetV2 + Grad-CAM)

An academic research prototype for dermoscopic skin-lesion classification across all 7 diagnostic categories of the **HAM10000** dataset, featuring deep transfer learning, group-leakage prevention, visual interpretability with **Grad-CAM**, and an interactive **Streamlit** dashboard.

> **ACADEMIC & NON-CLINICAL DISCLAIMER**  
> This system is built strictly for educational, scientific, and computer vision research purposes. It is **NOT** a medical diagnostic device, has not undergone clinical certification, and must never be used to diagnose, rule out, or guide medical therapy for skin malignancies. Always consult a board-certified dermatologist for clinical skin evaluation.

---

## 1. System Architecture & Features

- **Transfer Learning Backbone**: MobileNetV2 pretrained on ImageNet, utilizing inverted residual bottlenecks and depthwise separable convolutions for high inference efficiency.
- **Classification Head**: Global Average Pooling + Batch Normalization + Dropout (0.3) + Dense (128, ReLU) + Dropout (0.2) + 7-unit Softmax.
- **Canonical 7 Classes (HAM10000)**:
  1. `akiec` — Actinic Keratoses and Intraepithelial Carcinoma (Precancerous)
  2. `bcc` — Basal Cell Carcinoma (Malignant)
  3. `bkl` — Benign Keratosis-like Lesions (Benign)
  4. `df` — Dermatofibroma (Benign)
  5. `mel` — Melanoma (Critical Malignant)
  6. `nv` — Melanocytic Nevi (Benign Common Mole)
  7. `vasc` — Vascular Lesions (Benign)
- **Leakage Prevention**: Group-aware partitioning (`GroupShuffleSplit` on `lesion_id`) guarantees that multiple photos of the same physical lesion remain exclusively within either the training, validation, or test partition.
- **Class Balancing**: Balanced inverse class weights applied during categorical cross-entropy loss to counteract dataset imbalance (`nv` ~67%, `df` <2%).
- **Visual Explainability (Grad-CAM)**: Gradients of the predicted class score with respect to the `out_relu` convolutional activation map are pooled and weighted, generating a high-resolution attention heatmap blended onto the original image.
- **Modern Streamlit Dashboard**: Dark graphite aesthetic, 3-column Grad-CAM view, multi-class probability breakdown, model evaluation dashboard, and built-in demo lesion samples.

---

## 2. Directory Layout

```
skin_cancer_demo/
├── app.py                     # Streamlit web dashboard
├── train_model.py             # Transfer learning training script with leakage prevention
├── evaluate_model.py          # Standalone test-set evaluation & confusion matrix generator
├── gradcam.py                 # Modular Grad-CAM computation & overlay blending
├── dataset.py                 # Dataset discovery, canonical classes & demo sample generator
├── train_colab.ipynb          # Ready-to-run Google Colab GPU training notebook
├── requirements.txt           # Python dependencies
├── DEMO_GUIDE.md              # 3-minute presentation script & viva answers
├── PROJECT_STATUS.md          # Implementation status and system check log
├── TEST_REPORT.md             # Automated test execution report
├── data/
│   ├── HAM10000/              # (Optional) Full dataset location (metadata CSV + images)
│   └── sample_images/         # Demonstration dermoscopic images for testing
├── models/
│   ├── skin_lesion_model.keras# Trained Keras model weights
│   ├── classes.json           # Canonical 7-class label mapping
│   ├── model_config.json      # Model metadata and training configuration
│   ├── evaluation_report.json # Test metrics (accuracy, precision, recall, F1)
│   └── confusion_matrix.png   # Test set confusion matrix
├── results/                   # Curves, splits, and evaluation logs
└── tests/
    └── test_pipeline.py       # Automated test suite
```

---

## 3. Quick Start (Windows)

### Step 1: Initialize Virtual Environment & Dependencies
```powershell
# Open terminal in project directory
cd "d:\4-1\DL Project\skin_cancer_demo_starter\skin_cancer_demo"

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Step 2: (Optional) Verify via Pipeline Smoke Test
If the full 2.6 GB dataset is not yet downloaded, you can run a 1-epoch verification smoke test to verify model instantiation, weight saving, and Grad-CAM:
```powershell
python train_model.py --smoke_test
python evaluate_model.py
```

### Step 3: Launch Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 4. Full Dataset Training on HAM10000

### Option A: Local CPU/GPU Training
1. Download the HAM10000 dataset from [ISIC Archive](https://www.isic-archive.com/) or [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Extract the files under `data/HAM10000/`:
   ```
   data/HAM10000/
   ├── HAM10000_metadata.csv
   ├── HAM10000_images_part_1/ (*.jpg)
   └── HAM10000_images_part_2/ (*.jpg)
   ```
3. Run training:
   ```powershell
   python train_model.py --epochs 10 --batch_size 32
   python evaluate_model.py
   ```

### Option B: Google Colab GPU Acceleration (Recommended for Speed)
1. Open [Google Colab](https://colab.research.google.com/) and upload `train_colab.ipynb`.
2. Select **Runtime** > **Change runtime type** > **T4 GPU**.
3. Run all cells to download HAM10000, train MobileNetV2 with GPU acceleration (~10 minutes), evaluate, and download `trained_artifacts.zip`.
4. Extract the downloaded `models/` into your local `skin_cancer_demo/models/` folder.

---

## 5. Running Automated Tests
```powershell
python -m unittest tests/test_pipeline.py
```
This tests:
- Exact 7 canonical classes and index mapping.
- Metadata validation against corrupted or missing labels.
- Zero data leakage on `lesion_id` across splits.
- Demo sample generation.
