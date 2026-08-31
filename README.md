# MediScan — Deep Learning Medical Image Classification & Grad-CAM System

MediScan is an end-to-end AI-powered medical image classification platform designed to evaluate deep learning architectures (ResNet50 vs EfficientNet-B0), generate prediction confidence scores, visualize decision explainability with Grad-CAM heatmaps, and support bulk batch predictions.

---

## 🌟 Key Features

1. **Multi-Domain Medical Support:**
   - **Chest X-Ray Domain:** Normal, Bacterial Pneumonia, Viral Pneumonia, COVID-19.
   - **Skin Lesion Domain:** Benign Nevus, Melanoma, Seborrheic Keratosis.
2. **Transfer Learning Architectures:**
   - Pre-trained **ResNet50** and **EfficientNet-B0** backbones fine-tuned with custom classifier heads and Cosine Annealing learning rate schedulers.
3. **Grad-CAM Explainable AI:**
   - Spatial feature map attribution hooked into the final convolutional layers with customizable overlay opacity sliders and colormap palettes.
4. **Comprehensive Evaluation Suite:**
   - Real-time computation of Accuracy, Precision, Recall, F1-Score, Confusion Matrices, ROC Curves, and ROC-AUC scores.
5. **Architectural Benchmarking:**
   - Side-by-side comparative dashboard evaluating parameter efficiency, memory footprint, accuracy, and sub-second inference latency.
6. **Interactive Streamlit Web App:**
   - Modern glassmorphism dark-mode UI with single-image inference, batch processing with CSV exports, evaluation metrics, and live retraining controls.

---

## 🛠️ Project Structure

```text
mediscan/
├── config.py             # Hyperparameters, directory paths, and class labels
├── generate_dataset.py   # Synthetic medical dataset generator (Chest X-Ray & Skin Lesions)
├── dataset.py            # PyTorch Dataset, stratified splitting, and augmentation transforms
├── models.py             # ResNet50 and EfficientNet-B0 transfer learning wrappers
├── gradcam.py            # Custom PyTorch Grad-CAM explainability engine & heatmap blender
├── tracker.py            # Experiment logging engine (Structured JSON + MLflow)
├── train.py              # Model fine-tuning loop with AdamW and CosineAnnealingLR
├── evaluation.py         # Test evaluation metric suite & benchmark comparison generator
├── app.py                # Interactive Streamlit Web Application
├── MODEL_CARD.md         # Comprehensive model card & ethical disclaimers
└── README.md             # System documentation & usage guide
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install torch torchvision streamlit opencv-python matplotlib seaborn scikit-learn mlflow pillow albumentations
```

### 2. Generate Synthetic Dataset
```bash
python generate_dataset.py
```

### 3. Train Models
```bash
# Train ResNet50 on Chest X-Ray dataset
python train.py --task chest_xray --arch resnet50 --epochs 10

# Train EfficientNet-B0 on Chest X-Ray dataset
python train.py --task chest_xray --arch efficientnet_b0 --epochs 10
```

### 4. Run Side-by-Side Evaluation Benchmark
```bash
python evaluation.py
```

### 5. Launch Interactive Streamlit Web Application
```bash
python -m streamlit run app.py
```

---

## 📊 Evaluation & Success Metrics

- **Target Model Accuracy:** $> 85.0\%$
- **Target Inference Speed:** $< 3.0$ seconds / image
- **Explainability:** Grad-CAM spatial heatmap overlay enabled
- **Batch Processing:** Multi-image upload with downloadable CSV report

---

## ⚠️ Disclaimer
*MediScan is an educational and research machine learning project and is not intended to replace professional medical advice, diagnosis, or treatment.*
