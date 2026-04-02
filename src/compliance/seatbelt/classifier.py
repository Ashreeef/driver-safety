import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from typing import Tuple, Optional, Union, Dict

class SeatbeltClassifier:
    """
    MobileNetV3-Small classifier for seatbelt ON/OFF patches.
    This version strictly uses the PyTorch backend.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', config: Optional[Dict] = None):
        self.model_path = model_path
        self.device = device
        self.config = config or {}
        
        # Externalized settings
        self.input_size = tuple(self.config.get('input_size', [96, 96]))
        self.mean = np.array(self.config.get('normalization_mean', [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(self.config.get('normalization_std', [0.229, 0.224, 0.225]), dtype=np.float32)
        
        self._init_pytorch()
            
    def _init_pytorch(self):
        # Build architecture (MobileNetV3 Small)
        self.model = M.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, roi: np.ndarray) -> Tuple[int, float]:
        """
        Predict seatbelt status for a given ROI.
        Returns: (class_idx, confidence)
        """
        if roi is None or roi.size == 0:
            return 0, 0.5
            
        # Preprocessing (matches notebook val_transforms)
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, self.input_size).astype(np.float32) / 255.0
        
        # Normalization
        img = (img - self.mean) / self.std
        
        # Batch and channel shuffle
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            
        return int(probs.argmax().item()), float(probs[1].item())

def build_patch_cnn(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Legacy helper for building the model architecture for training."""
    weights = M.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = M.mobilenet_v3_small(weights=weights)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m
