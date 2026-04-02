import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Union

class EMASmoother:
    """
    Exponential Moving Average smoother for prediction probabilities.
    """
    def __init__(self, alpha: float = 0.35, threshold: float = 0.52, min_frames: int = 6):
        self.alpha = alpha
        self.threshold = threshold
        self.min_frames = min_frames
        self.ema = 0.5
        self.n = 0

    def reset(self):
        self.ema = 0.5
        self.n = 0

    def update(self, on_prob: float) -> Tuple[str, float]:
        self.ema = self.alpha * on_prob + (1.0 - self.alpha) * self.ema
        self.n += 1
        
        if self.n < self.min_frames:
            return 'WARMING', self.ema
            
        return ('ON' if self.ema >= self.threshold else 'OFF'), self.ema

class MajorityVoteSmoother:
    """
    Majority vote smoother over a rolling window.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def reset(self):
        self.history.clear()

    def update(self, cls_idx: int, conf: float) -> Tuple[int, float]:
        self.history.append((cls_idx, conf))
        
        # Majority vote
        votes = [x[0] for x in self.history]
        confs = [x[1] for x in self.history]
        
        sm_cls = max(set(votes), key=votes.count)
        sm_conf = sum(confs) / len(confs)
        
        return sm_cls, sm_conf

class FusionBiLSTM(nn.Module):
    """
    BiLSTM temporal fusion model.
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.proj(x)
        out, _ = self.bilstm(x)
        # We take the last hidden state for prediction
        return self.head(out[:, -1, :])

class BiLSTMSmoother:
    """
    Wrapper for BiLSTM model to perform sequence-based smoothing.
    """
    def __init__(self, model_path: str, sequence_len: int = 16, device: str = 'cpu'):
        self.device = device
        self.sequence_len = sequence_len
        self.model = FusionBiLSTM()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.buffer = deque(maxlen=sequence_len)

    def reset(self):
        self.buffer.clear()

    def update(self, fused_vector: List[float]) -> Tuple[Optional[int], float]:
        self.buffer.append(fused_vector)
        
        if len(self.buffer) < self.sequence_len:
            return None, 0.0
            
        seq = torch.from_numpy(np.array(list(self.buffer), dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(seq)
            probs = F.softmax(logits, dim=1)[0]
            
        return int(probs.argmax().item()), float(probs[1].item())
