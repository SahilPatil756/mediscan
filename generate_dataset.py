import os
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def generate_chest_xray(class_name, size=(224, 224)):
    """Generate synthetic Chest X-Ray image with realistic anatomical features and pathology indicators."""
    h, w = size
    img = np.zeros((h, w), dtype=np.uint8) + 20  # dark background
    
    # 1. Torso contour (broad gray area)
    cv2.ellipse(img, (w // 2, h // 2 + 10), (w // 2 - 15, h // 2 - 10), 0, 0, 360, 110, -1)
    
    # 2. Lung fields (dark bilateral cavities)
    cv2.ellipse(img, (w // 2 - 42, h // 2 - 5), (28, 65), -10, 0, 360, 35, -1)
    cv2.ellipse(img, (w // 2 + 42, h // 2 - 5), (28, 65), 10, 0, 360, 35, -1)
    
    # 3. Spine & Rib structures
    cv2.line(img, (w // 2, 20), (w // 2, h - 20), 160, 4)
    for y in range(40, h - 30, 22):
        cv2.ellipse(img, (w // 2 - 35, y), (35, 12), 15, 180, 360, 120, 2)
        cv2.ellipse(img, (w // 2 + 35, y), (35, 12), -15, 180, 360, 120, 2)
        
    # 4. Heart shadow (left lower lung region, slightly brighter)
    cv2.ellipse(img, (w // 2 + 15, h // 2 + 25), (25, 35), -20, 0, 360, 140, -1)
    
    # 5. Clavicles
    cv2.line(img, (w // 2 - 70, 45), (w // 2 - 10, 55), 170, 3)
    cv2.line(img, (w // 2 + 70, 45), (w // 2 + 10, 55), 170, 3)

    # Blur for X-ray softness
    img = cv2.GaussianBlur(img, (9, 9), 0)
    
    # 6. Pathology specific features
    if class_name == "Bacterial Pneumonia":
        # Focal dense opacity in right lower lung
        center = (w // 2 - 40 + np.random.randint(-5, 5), h // 2 + 15 + np.random.randint(-5, 5))
        cv2.circle(img, center, np.random.randint(16, 25), (200), -1)
        img = cv2.GaussianBlur(img, (15, 15), 0)
        
    elif class_name == "Viral Pneumonia":
        # Diffuse patchy infiltrates across both lungs
        for _ in range(8):
            px = np.random.randint(w // 2 - 55, w // 2 + 55)
            py = np.random.randint(h // 2 - 40, h // 2 + 35)
            cv2.circle(img, (px, py), np.random.randint(8, 16), (160), -1)
        img = cv2.GaussianBlur(img, (13, 13), 0)
        
    elif class_name == "COVID-19":
        # Peripheral bilateral ground glass opacities
        for px, py in [(w // 2 - 58, h // 2 - 10), (w // 2 - 55, h // 2 + 25),
                       (w // 2 + 58, h // 2 - 5), (w // 2 + 55, h // 2 + 20)]:
            cv2.ellipse(img, (px, py), (16, 28), np.random.randint(-15, 15), 0, 360, (180), -1)
        img = cv2.GaussianBlur(img, (17, 17), 0)

    # Add Poisson/Gaussian noise
    noise = np.random.normal(0, 5, (h, w)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Convert grayscale to 3-channel RGB image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_rgb)


def generate_skin_lesion(class_name, size=(224, 224)):
    """Generate synthetic Skin Lesion image with dermatological patterns."""
    h, w = size
    # 1. Base skin tone (pale brown/pink)
    skin_color = np.array([np.random.randint(210, 240), np.random.randint(180, 210), np.random.randint(165, 195)], dtype=np.uint8)
    img = np.full((h, w, 3), skin_color, dtype=np.uint8)
    
    # Texture noise
    skin_noise = np.random.normal(0, 6, (h, w, 3)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + skin_noise, 0, 255).astype(np.uint8)
    
    cx, cy = w // 2 + np.random.randint(-10, 10), h // 2 + np.random.randint(-10, 10)
    
    if class_name == "Benign Nevus":
        # Smooth symmetrical circular/oval lesion with uniform dark brown color
        r = np.random.randint(30, 45)
        lesion_color = (np.random.randint(60, 90), np.random.randint(40, 65), np.random.randint(25, 45))
        cv2.circle(img, (cx, cy), r, lesion_color, -1)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        
    elif class_name == "Melanoma":
        # Asymmetrical, irregular border, multi-color lesion (dark, reddish, black)
        pts = []
        num_pts = 12
        base_r = np.random.randint(35, 55)
        for i in range(num_pts):
            angle = i * (2 * np.pi / num_pts)
            r = base_r + np.random.randint(-15, 20)
            pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        
        # Base irregular polygon
        cv2.fillPoly(img, [pts], (20, 20, 20))
        # Inner reddish/brown irregular patches
        cv2.circle(img, (cx + 5, cy - 5), base_r // 2, (90, 30, 25), -1)
        cv2.circle(img, (cx - 8, cy + 8), base_r // 3, (140, 40, 30), -1)
        img = cv2.GaussianBlur(img, (9, 9), 0)
        
    elif class_name == "Seborrheic Keratosis":
        # Stuck-on warty lesion with yellowish-brown tones and keratin dots
        r_x, r_y = np.random.randint(35, 50), np.random.randint(28, 42)
        lesion_color = (np.random.randint(110, 140), np.random.randint(80, 110), np.random.randint(40, 70))
        cv2.ellipse(img, (cx, cy), (r_x, r_y), np.random.randint(0, 180), 0, 360, lesion_color, -1)
        
        # Keratin cysts (tiny white/yellow dots)
        for _ in range(12):
            kx = cx + np.random.randint(-r_x + 8, r_x - 8)
            ky = cy + np.random.randint(-r_y + 8, r_y - 8)
            cv2.circle(img, (kx, ky), 2, (220, 210, 160), -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
    return Image.fromarray(img)


def build_dataset(task="chest_xray", num_samples_per_class=120):
    """Build train/val/test directory splits for task dataset."""
    set_seed(42)
    classes = config.CLASS_NAMES[task]
    base_dir = config.CHEST_XRAY_DIR if task == "chest_xray" else config.SKIN_LESION_DIR
    
    print(f"Generating synthetic dataset for task: '{task}' with classes {classes}...")
    
    for split in ["train", "val", "test"]:
        for c in classes:
            (base_dir / split / c).mkdir(parents=True, exist_ok=True)
            
    for class_name in classes:
        # Generate samples
        samples = []
        for i in range(num_samples_per_class):
            if task == "chest_xray":
                img = generate_chest_xray(class_name)
            else:
                img = generate_skin_lesion(class_name)
            samples.append(img)
            
        # Stratified 70/15/15 split
        n_total = len(samples)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]
        
        splits = [("train", train_samples), ("val", val_samples), ("test", test_samples)]
        for split_name, split_imgs in splits:
            split_dir = base_dir / split_name / class_name
            for idx, img in enumerate(split_imgs):
                filepath = split_dir / f"{class_name.lower().replace(' ', '_')}_{idx+1:04d}.png"
                img.save(filepath)
                
    print(f"Dataset generated successfully at {base_dir}!")


if __name__ == "__main__":
    build_dataset("chest_xray", num_samples_per_class=120)
    build_dataset("skin_lesion", num_samples_per_class=100)
