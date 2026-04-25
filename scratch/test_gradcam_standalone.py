import torch
import torch.nn as nn
import numpy as np
from src.utils.gradcam import GradCAM
import torchvision.models as M

def test_gradcam():
    print("Testing Grad-CAM...")
    model = M.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.eval()
    
    target_layer = model.features[12]
    grad_cam = GradCAM(model, target_layer)
    
    input_tensor = torch.randn(1, 3, 96, 96, requires_grad=True)
    
    try:
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx=1)
        print(f"Heatmap generated successfully. Shape: {heatmap.shape}")
        print(f"Heatmap max: {heatmap.max()}, min: {heatmap.min()}")
        if heatmap.max() > 0:
            print("Grad-CAM is WORKING!")
        else:
            print("Grad-CAM returned all zeros.")
    except Exception as e:
        print(f"Grad-CAM failed with error: {e}")

if __name__ == "__main__":
    test_gradcam()
