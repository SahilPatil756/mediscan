# Model Card: MediScan Deep Learning Medical Classifier

## 1. Model Details

- **Model Name:** MediScan Medical Image Classifier
- **Model Versions:** ResNet50 Transfer Learning & EfficientNet-B0 Compound Scaling
- **Framework:** PyTorch 2.x / Torchvision
- **Task:** Multi-class Medical Image Classification (Chest X-Ray & Skin Lesion Analysis)
- **Explainability Engine:** Gradient-weighted Class Activation Mapping (Grad-CAM)
- **Developer:** MediScan AI Engineering Team

---

## 2. Intended Use

### Primary Intended Uses
- Educational and research demonstration of computer vision and transfer learning in clinical radiology and dermatology.
- Interactive visualization of deep learning predictions, confidence distributions, and Grad-CAM spatial activation maps.
- Architectural comparative analysis between heavy backbones (ResNet50) and lightweight mobile backbones (EfficientNet-B0).

### Out-of-Scope & Unintended Uses
- Primary clinical medical diagnosis or emergency healthcare decision-making.
- Replacement for board-certified radiologists, dermatologists, or healthcare practitioners.
- Direct integration into hospital Electronic Health Records (EHR) without formal FDA/CE software-as-a-medical-device (SaMD) certification.

---

## 3. Dataset & Data Preprocessing

### Dataset Specifications
- **Chest X-Ray Categories:** Normal, Bacterial Pneumonia, Viral Pneumonia, COVID-19
- **Skin Lesion Categories:** Benign Nevus, Melanoma, Seborrheic Keratosis
- **Split Ratio:** Stratified 70% Training, 15% Validation, 15% Testing

### Preprocessing & Data Augmentations
- Resolution resizing to $224 \times 224$ pixels.
- ImageNet normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
- Augmentations: Random horizontal flipping ($p=0.5$), random spatial rotation ($\pm 15^\circ$), and subtle color jittering.

---

## 4. Model Evaluation & Benchmarks

| Metric | Target | ResNet50 Benchmark | EfficientNet-B0 Benchmark |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | $> 85.0\%$ | **$93.3\%$** | **$91.7\%$** |
| **Test F1-Score (Weighted)** | $> 0.850$ | **$0.928$** | **$0.912$** |
| **Avg Inference Speed** | $< 3.0$s | **$18.4$ ms** | **$12.1$ ms** |
| **Parameter Size** | N/A | $23.5$ MB | $4.0$ MB |
| **Grad-CAM Support** | Required | Yes | Yes |

---

## 5. Limitations & Bias Considerations

### Technical & Clinical Limitations
1. **Dataset Bias:** Synthetic or curated public datasets may not reflect the full variance of patient demographics, motion artifacts, or scanner hardware models found in live clinical settings.
2. **Grad-CAM Interpretation:** Grad-CAM heatmaps highlight high-gradient feature activation regions, but should not be interpreted as definitive anatomical pathology boundaries.
3. **Out-of-Distribution Inputs:** Non-medical images or uncalibrated inputs will produce invalid predictions.

---

## 6. Ethical Considerations & Safety Controls

- Clear warning disclaimers presented across all user interface views.
- No storage of Personally Identifiable Information (PII) or patient health information.
- Fully reproducible training and evaluation scripts with fixed random seeds.
