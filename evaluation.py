import time
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, roc_auc_score
)

import config
import dataset
import models

def evaluate_model(model, test_loader, class_names, device=config.DEVICE):
    """
    Evaluate trained model on test dataset split.
    
    Returns:
        metrics dict: accuracy, precision, recall, f1, confusion matrix, roc data, latency
    """
    model.eval()
    model.to(device)
    
    y_true = []
    y_pred = []
    y_scores = []
    latencies = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            
            # Measure inference latency for batch
            t0 = time.time()
            outputs = model(images)
            t1 = time.time()
            
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_scores.extend(probs)
            latencies.append((t1 - t0) / images.size(0))
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate ROC-AUC (One-vs-Rest for multi-class)
    roc_dict = {}
    try:
        num_classes = len(class_names)
        y_true_onehot = np.eye(num_classes)[y_true]
        macro_auc = roc_auc_score(y_true_onehot, y_scores, average="macro", multi_class="ovr")
        
        for idx, c_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, idx], y_scores[:, idx])
            class_auc = auc(fpr, tpr)
            roc_dict[c_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(class_auc)}
    except Exception as e:
        print(f"ROC AUC computation notice: {e}")
        macro_auc = 0.0
        
    avg_latency_ms = np.mean(latencies) * 1000.0
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc_macro": float(macro_auc),
        "avg_latency_ms": float(avg_latency_ms),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            c_name: {
                "precision": float(per_class_p[idx]),
                "recall": float(per_class_r[idx]),
                "f1": float(per_class_f1[idx])
            } for idx, c_name in enumerate(class_names)
        },
        "roc_curves": roc_dict
    }
    return metrics


def generate_evaluation_plots(metrics, class_names, architecture="resnet50", task="chest_xray"):
    """Generate and save PNG plots for Confusion Matrix and ROC Curves."""
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = np.array(metrics["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {architecture.upper()} ({task})")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    cm_path = results_dir / f"confusion_matrix_{architecture}_{task}.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    
    # 2. ROC Curves Plot
    plt.figure(figsize=(7, 5))
    for c_name, roc_data in metrics.get("roc_curves", {}).items():
        plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"{c_name} (AUC = {roc_data['auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.title(f"ROC Curves — {architecture.upper()} ({task})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = results_dir / f"roc_curves_{architecture}_{task}.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    
    return str(cm_path), str(roc_path)


def compare_models(task="chest_xray"):
    """
    Run evaluation on both ResNet50 and EfficientNet-B0 and return comparative benchmark.
    """
    print(f"--- Running Comparative Evaluation for Task: '{task}' ---")
    _, _, test_loader, class_names = dataset.get_dataloaders(task=task)
    num_classes = len(class_names)
    
    architectures = ["resnet50", "efficientnet_b0"]
    comparison = {}
    
    for arch in architectures:
        ckpt_path = config.MODELS_DIR / f"{arch}_{task}.pt"
        model = models.get_model(architecture=arch, num_classes=num_classes, pretrained=False)
        
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint for {arch} (Val Acc: {checkpoint.get('val_acc', 0.0)*100:.1f}%)")
        else:
            print(f"No checkpoint found for {arch}. Running evaluation on initialized pre-trained weights.")
            model = models.get_model(architecture=arch, num_classes=num_classes, pretrained=True)
            
        metrics = evaluate_model(model, test_loader, class_names)
        generate_evaluation_plots(metrics, class_names, architecture=arch, task=task)
        
        # Count total trainable & total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        comparison[arch] = {
            "metrics": metrics,
            "total_params": total_params,
            "param_size_mb": round(total_params * 4 / (1024 * 1024), 2)
        }
        
    # Save benchmark JSON
    benchmark_path = config.RESULTS_DIR / f"comparison_{task}.json"
    with open(benchmark_path, "w") as f:
        json.dump(comparison, f, indent=2)
        
    print(f"Model comparison saved to {benchmark_path}")
    return comparison


if __name__ == "__main__":
    compare_models("chest_xray")
