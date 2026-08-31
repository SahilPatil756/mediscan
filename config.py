import os
from pathlib import Path
import torch

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHEST_XRAY_DIR = DATA_DIR / "chest_xray"
SKIN_LESION_DIR = DATA_DIR / "skin_lesion"

# Output Directories
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHEST_XRAY_DIR, SKIN_LESION_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model & Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 12
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# ImageNet Normalization Stats
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Dataset Task Settings
DEFAULT_TASK = "chest_xray"

CLASS_NAMES = {
    "chest_xray": ["Normal", "Bacterial Pneumonia", "Viral Pneumonia", "COVID-19"],
    "skin_lesion": ["Benign Nevus", "Melanoma", "Seborrheic Keratosis"]
}

# Target Accuracy Threshold
TARGET_ACCURACY = 0.85
