import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights

class ResNet50Classifier(nn.Module):
    """ResNet50 model wrapper with custom classification head for medical images."""
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Replace classifier fc layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def freeze_backbone(self):
        """Freeze feature extraction layers, leaving only classifier head trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_last_layer(self):
        """Unfreeze layer4 and classifier for fine-tuning."""
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 model wrapper with custom classification head for medical images."""
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def freeze_backbone(self):
        """Freeze feature extraction layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
            
    def unfreeze_last_layer(self):
        """Unfreeze top feature blocks and classifier."""
        for param in self.backbone.features[-2:].parameters():
            param.requires_grad = True
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True


def get_model(architecture="resnet50", num_classes=4, pretrained=True):
    """Factory function to build and return requested deep learning model architecture."""
    architecture = architecture.lower()
    if architecture == "resnet50":
        model = ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)
    elif architecture in ["efficientnet_b0", "efficientnet"]:
        model = EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: 'resnet50', 'efficientnet_b0'")
        
    return model
