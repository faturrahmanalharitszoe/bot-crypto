import torch
import torch.nn as nn

class DayTradingMLP(nn.Module):
    """
    Lightweight PyTorch MLP classifier for crypto trading signals.
    Inputs: 25 MTF technical features.
    Outputs: Logits for 2 classes: Class 0 (LONG) and Class 1 (SHORT).
    
    Includes Batch Normalization and Dropout to regularize and stabilize learning.
    Designed to run efficiently (<5ms) on a CPU-only environment.
    """
    def __init__(self, input_dim: int = 25, hidden_dims: list[int] = [64, 32], dropout: float = 0.2, activation: str = "relu"):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            
            act_name = activation.lower()
            if act_name == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif act_name == "mish":
                layers.append(nn.Mish())
            else:
                layers.append(nn.ReLU())
                
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        # Final layer outputs logits for 2 classes:
        # Class 0 (maps to original class 1: LONG)
        # Class 1 (maps to original class 2: SHORT)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

