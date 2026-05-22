import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM implementation for PyTorch CNNs.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        # Save training state and force train mode to bypass internal inference-mode restrictions
        was_training = self.model.training
        self.model.train() 
        
        # Ensure BatchNorm and Dropout behave like eval mode
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)):
                m.eval()

        try:
            with torch.set_grad_enabled(True):
                output = self.model(input_tensor)
                
                # Handle cases where output might be a list (like YOLO) or a tensor
                if isinstance(output, (list, tuple)):
                    output = output[0]
                    
                if isinstance(output, dict):
                    if 'scores' in output:
                        # YOLOv8 in train mode returns dict with 'scores' shape [batch, num_classes, num_anchors]
                        score = output['scores'][0, class_idx].max()
                    else:
                        raise ValueError(f"Unexpected dict keys from model: {output.keys()}")
                elif output.dim() == 3: # YOLO-style (batch, items, anchors)
                    # Take max score for class_idx across all anchors
                    score = output[0, 4 + class_idx].max()
                else:
                    score = output[0, class_idx]
                
                self.model.zero_grad()
                score.backward(retain_graph=True)
                
                if self.gradients is None or self.activations is None:
                    # Fallback to uniform if hooks failed
                    return np.ones((input_tensor.shape[2]//32, input_tensor.shape[3]//32), dtype=np.float32)

                gradients = self.gradients.detach()
                activations = self.activations.detach()
                
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * activations, dim=1).squeeze()
                
                cam = F.relu(cam)
                cam = cam.cpu().numpy()
                
                if cam.ndim == 3: # multi-scale case
                    cam = cam[0] # take first scale

                if cam.max() > 0:
                    cam = cam / cam.max()
                    
                return cam
        finally:
            self.model.train(was_training)

class YOLOGradCAM:
    """
    Grad-CAM wrapper for Ultralytics YOLO models.
    """
    def __init__(self, yolo_model, target_layer_idx: int = 9):
        # yolo_model is an ultralytics.YOLO or torch.hub.load instance
        if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'model'):
            self.model = yolo_model.model
            # For ultralytics YOLOv8 or YOLOv5 torch.hub with nested model
            self.layers = self.model.model
        elif hasattr(yolo_model, 'model'):
            self.model = yolo_model.model
            self.layers = self.model
        else:
            self.model = yolo_model
            self.layers = getattr(self.model, 'model', self.model)

        self.target_layer = self.layers[target_layer_idx]
        self.grad_cam = GradCAM(self.model, self.target_layer)

    def get_heatmap(self, frame: np.ndarray, class_idx: int = 1) -> np.ndarray:
        # Preprocessing for YOLO (standard 640x640 or as configured)
        H, W = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img = img.transpose(2, 0, 1) # BGR to CHW
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        tensor = tensor.to(next(self.model.parameters()).device)
        tensor.requires_grad = True
        
        heatmap = self.grad_cam.generate_heatmap(tensor, class_idx=class_idx)
        heatmap = cv2.resize(heatmap, (W, H))
        return heatmap
