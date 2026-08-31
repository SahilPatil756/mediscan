import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import config

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for model explainability."""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        if self.target_layer is None:
            self.target_layer = self._find_target_layer()
            
        self.handles = []
        self._register_hooks()
        
    def _find_target_layer(self):
        """Locate suitable target layer automatically based on architecture."""
        if hasattr(self.model, "backbone"):
            b = self.model.backbone
            if hasattr(b, "layer4"):  # ResNet50
                return b.layer4[-1]
            elif hasattr(b, "features"):  # EfficientNet-B0
                return b.features[-1]
        raise ValueError("Could not find default target layer for Grad-CAM. Please specify explicitly.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
            
        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for given input tensor and target class.
        
        Args:
            input_tensor: PyTorch tensor [1, C, H, W]
            target_class: Integer index of target class. If None, highest predicted class is used.
            
        Returns:
            heatmap: Normalized 2D numpy array [H, W] in [0.0, 1.0]
            predicted_class: Integer class index
            confidence: Float confidence score
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = torch.argmax(probabilities, dim=1).item()
            
        confidence = probabilities[0, target_class].item()
        
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Calculate weights alpha = Mean(Gradient over H and W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of feature activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize heatmap to match input tensor spatial dimension (H, W)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize heatmap to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class, confidence
        
    def overlay_heatmap(self, original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap onto RGB original image.
        
        Args:
            original_img: PIL Image or RGB numpy array [H, W, 3]
            heatmap: 2D float numpy array [H, W] in [0, 1]
            alpha: Blend weight float (0.0 to 1.0)
            colormap: OpenCV colormap constant
            
        Returns:
            overlay_rgb: RGB numpy array [H, W, 3]
            heatmap_rgb: RGB heatmap numpy array [H, W, 3]
        """
        if isinstance(original_img, Image.Image):
            orig_np = np.array(original_img)
        else:
            orig_np = original_img.copy()
            
        h, w = orig_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply OpenCV colormap (returns BGR)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        # Overlay original image and heatmap
        overlay_rgb = cv2.addWeighted(orig_np, 1 - alpha, heatmap_rgb, alpha, 0)
        return overlay_rgb, heatmap_rgb
        
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
