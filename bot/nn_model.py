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
    def __init__(self, input_dim: int = 31, hidden_dims: list[int] = [64, 32], dropout: float = 0.2, activation: str = "relu"):
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


class DayTradingCLSTM(nn.Module):
    """
    Hybrid C-LSTM (CNN + LSTM) classifier for crypto trading signals.
    Features: 38 inputs (31 base/MTF + 7 Fibonacci retracement features).
    Sequence Length: 10 historical candles.
    Inputs: [batch_size, seq_len, input_dim]
    Outputs: Logits for 2 classes (LONG vs SHORT).
    
    Structure:
      1. Conv1D over features to extract local temporal patterns.
      2. LSTM to model long-term temporal dependencies.
      3. Fully Connected layers to map hidden states to direction logits.
    """
    def __init__(self, input_dim: int = 38, seq_len: int = 10, num_classes: int = 2,
                 cnn_channels: int = 32, lstm_hidden: int = 64, cnn_kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.dropout = dropout

        # Conv1D expects shape: [batch, input_dim, seq_len]
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM expects shape: [batch, seq_len, cnn_channels]
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        # Transpose for Conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Transpose back for LSTM: [batch_size, seq_len, cnn_channels]
        x = x.transpose(1, 2)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take the output of the last sequence step
        out = lstm_out[:, -1, :] # [batch_size, lstm_hidden]
        
        logits = self.fc(out)
        return logits


