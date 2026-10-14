# College Demonstration & Presentation Guide
## Explainable Skin Cancer Detection Using CNN

---

### 1. Project Overview & Quick Summary
- **Title**: Explainable Skin Cancer Detection Using Convolutional Neural Networks (CNN) & Grad-CAM
- **Dataset**: HAM10000 ("Human Against Machine with 10,000 training images"), containing 10,015 dermoscopic photographs across 7 diagnostic categories.
- **Backbone Architecture**: MobileNetV2 pretrained on ImageNet with a customized 7-class transfer learning head.
- **Interpretability**: Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional activation layer (`out_relu`).
- **User Interface**: Streamlit web application with dark graphite medical theme and real-time inference & Grad-CAM heatmap overlay.

---

### 2. 3-Minute Live Demonstration Flow

#### Minute 1: Introduction & Architecture Setup
1. Launch the Streamlit application:
   ```powershell
   .\.venv\Scripts\streamlit.exe run app.py
   ```
2. Explain the dashboard:
   - "This is an academic research demonstration for computer vision in dermatology."
   - Point to the **Model Details** in the sidebar: MobileNetV2 backbone, 224×224 resolution, 7 HAM10000 classes.

#### Minute 2: Dermoscopic Image Analysis & Predictions
1. Select a sample lesion from the dropdown (e.g. `melanoma_sample_mel.jpg` or `nevus_sample_nv.jpg`), or upload an image.
2. Highlight the **Prediction Panel**:
   - Note the predicted diagnosis, full clinical name, risk category (e.g., *Malignant* vs *Benign*), and confidence score.
   - Expand the **View All 7 Diagnostic Category Probabilities** table and bar chart to show the model computes a full probability distribution, not just a single binary output.

#### Minute 3: Grad-CAM Explainability & Academic Boundaries
1. Focus on the **3-column Grad-CAM display**:
   - Column 1: Preprocessed Input Image.
   - Column 2: Raw Grad-CAM Activation Heatmap.
   - Column 3: Blended Overlay on the Original Lesion.
2. Adjust the **Heatmap Blending Opacity** slider or switch the **Colormap** (e.g., from `jet` to `turbo` or `inferno`) in the sidebar.
3. State the key interpretability takeaway:
   - *"Grad-CAM highlights regions with high positive gradients that influenced the model's classification score. This allows clinicians to audit whether the network detected authentic pigment network irregularities or if it was falsely triggered by artifacts such as surgical pen markings, air bubbles, or skin hairs."*
4. Click on the **Model Evaluation & Metrics** tab:
   - Show the held-out test confusion matrix and per-class precision/recall/F1-score table.
   - Emphasize that the splits were generated with **Group-Aware Splitting (`lesion_id`)** to prevent multiple photos of the same lesion leaking between training and test sets.

---

### 3. Key Viva / Presentation Questions & Winning Answers

#### Q1: "Why did you choose MobileNetV2 over a deeper network like ResNet50 or VGG16?"
> **Answer**: *"MobileNetV2 uses depthwise separable convolutions and inverted residual bottlenecks. It achieves comparable top-1 accuracy to ResNet-50 with ~3.4 million parameters instead of 25+ million, reducing inference latency by 4x. This makes it ideal for real-time edge deployment and web-based clinical demonstration without sacrificing feature richness."*

#### Q2: "What is data leakage and how did you prevent it in HAM10000?"
> **Answer**: *"The HAM10000 dataset includes multiple dermatoscopic photographs of the same physical lesion taken from slightly different angles or follow-up visits. A standard random train/test split would place images of the same lesion in both sets, causing optimistic overfitting. We implemented `GroupShuffleSplit` on `lesion_id`, guaranteeing completely disjoint patient lesions across train, validation, and test sets."*

#### Q3: "How does Grad-CAM work mathematically?"
> **Answer**: *"Grad-CAM computes the gradient of the predicted class score $y^c$ with respect to the feature map activations $A^k$ of the final convolutional layer (`out_relu`). We compute neuron importance weights $\alpha_k^c$ via global average pooling of these gradients. Finally, we take a weighted combination of feature maps followed by a ReLU activation to strictly retain features with a positive influence on the target class."*

#### Q4: "How do you handle class imbalance in HAM10000?"
> **Answer**: *"HAM10000 is heavily imbalanced; melanocytic nevi (`nv`) make up over 67% of cases, while dermatofibroma (`df`) is less than 2%. We applied inverse class-frequency weighting during cross-entropy optimization using scikit-learn's `compute_class_weight('balanced')`, ensuring rare but critical malignancies like melanoma and dermatofibroma are penalized appropriately during training."*

#### Q5: "Can this system be used directly in a hospital?"
> **Answer**: *"No. This is strictly an academic research prototype. Clinical deployment requires prospective multi-center randomized clinical trials, regulatory clearance (FDA 510(k) or CE marking), robustness validation across diverse Fitzpatrick skin types, and integration with histopathological biopsy workflows."*

---

### 4. Canonical HAM10000 7-Class Reference
| Code | Medical Category | Clinical Risk |
|---|---|---|
| `akiec` | Actinic Keratoses & Intraepithelial Carcinoma | Precancerous / Early Malignant |
| `bcc` | Basal Cell Carcinoma | Malignant (Locally destructive) |
| `bkl` | Benign Keratosis-like Lesions | Benign (Harmless) |
| `df` | Dermatofibroma | Benign (Fibrous nodule) |
| `mel` | Melanoma | Critical Malignant (High metastasis risk) |
| `nv` | Melanocytic Nevi (Common moles) | Benign |
| `vasc` | Vascular Lesions | Benign |
