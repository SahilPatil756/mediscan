import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
import dataset
import models
from tracker import ExperimentTracker

def train_model(task="chest_xray", architecture="resnet50", epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE, batch_size=config.BATCH_SIZE):
    """
    Train and fine-tune selected model architecture on task dataset.
    """
    device = config.DEVICE
    print(f"--- Launching Training: Task='{task}', Model='{architecture}', Device='{device}' ---")
    
    # 1. Load Data
    train_loader, val_loader, test_loader, class_names = dataset.get_dataloaders(task=task, batch_size=batch_size)
    num_classes = len(class_names)
    
    # 2. Instantiate Model
    model = models.get_model(architecture=architecture, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Two-stage strategy: start with frozen backbone, then unfreeze top layers after 2 epochs
    model.freeze_backbone()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Stage 2: Unfreeze top layers after epoch 2 for fine-tuning
        if epoch == 3:
            print("Unfreezing top feature layers for fine-tuning...")
            model.unfreeze_last_layer()
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 0.5, weight_decay=config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 2, eta_min=1e-6)
            
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        scheduler.step()
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")
        
        # Save Best Model Checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_path = config.MODELS_DIR / f"{architecture}_{task}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": architecture,
                "task": task,
                "class_names": class_names,
                "val_acc": best_val_acc,
                "history": history
            }, save_path)
            
    training_duration = time.time() - start_time
    print(f"Training Complete in {training_duration:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%")
    
    # Log Experiment
    tracker = ExperimentTracker()
    run_id = f"{architecture}_{task}_{int(time.time())}"
    params = {
        "task": task,
        "architecture": architecture,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR"
    }
    metrics = {
        "best_val_acc": round(best_val_acc, 4),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "training_duration_sec": round(training_duration, 1)
    }
    tracker.log_run(run_id, params, metrics, history)
    return best_val_acc, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MediScan Deep Learning Classifiers")
    parser.add_argument("--task", type=str, default="chest_xray", choices=["chest_xray", "skin_lesion"])
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    train_model(task=args.task, architecture=args.arch, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
