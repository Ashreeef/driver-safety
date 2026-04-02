import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from typing import Tuple, Optional, Union

class SeatbeltClassifier:
    """
    MobileNetV3-Small classifier for seatbelt ON/OFF patches.
    Supports both PyTorch and ONNX Runtime backends.
    """
    
    def __init__(self, model_path: str, backend: str = 'pytorch', device: str = 'cpu'):
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.model = None
        self.session = None
        
        if backend == 'pytorch':
            self._init_pytorch()
        elif backend == 'onnx':
            self._init_onnx()
            
    def _init_pytorch(self):
        # Build architecture (MobileNetV3 Small)
        self.model = M.mobilenet_v3_small(weights=None)
        # Match the head from the notebook: m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _init_onnx(self):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        except ImportError:
            raise ImportError("onnxruntime is required for the ONNX backend.")

    def predict(self, roi: np.ndarray) -> Tuple[int, float]:
        """
        Predict seatbelt status for a given ROI.
        Returns: (class_idx, confidence)
        """
        if roi is None or roi.size == 0:
            return 0, 0.5
            
        if self.backend == 'pytorch':
            return self._predict_pytorch(roi)
        else:
            return self._predict_onnx(roi)

    def _predict_pytorch(self, roi: np.ndarray) -> Tuple[int, float]:
        # Preprocessing (matches notebook val_transforms)
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (96, 96)).astype(np.float32) / 255.0
        
        # Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Batch and channel shuffle
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            
        return int(probs.argmax().item()), float(probs[1].item())

    def _predict_onnx(self, roi: np.ndarray) -> Tuple[int, float]:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (96, 96)).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        t = ((img - mean) / std).transpose(2, 0, 1)[None]
        
        logits = self.session.run(None, {'input': t})[0]
        # Softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        
        return int(probs[0].argmax()), float(probs[0, 1])

def build_patch_cnn(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Legacy helper for building the model architecture for training."""
    weights = M.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = M.mobilenet_v3_small(weights=weights)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m
