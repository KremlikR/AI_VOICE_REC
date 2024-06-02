import torch
import torch.nn as nn
from torch.nn import functional as F
from script.ActDropNormCNN1D import ActDropNormCNN1D

class SpeechRecognition(nn.Module):
    """
    Speech Recognition model with CNN, Dense, and LSTM layers.
    """
    hyper_parameters = {
        "num_classes": 29,
        "n_feats": 81,
        "dropout": 0.1,
        "hidden_size": 1024,
        "num_layers": 1
    }

    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, kernel_size=10, stride=2, padding=10//2),
            ActDropNormCNN1D(n_feats, dropout),
        )
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden_size,
            num_layers=num_layers, dropout=0.0,
            bidirectional=False
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def _init_hidden(self, batch_size):
        """
        Initialize the hidden state for LSTM.
        """
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n, batch_size, hs), torch.zeros(n, batch_size, hs))

    def forward(self, x, hidden):
        """
        Forward pass for the model.
        """
        x = x.squeeze(1)  # (batch, feature, time)
        x = self.cnn(x)  # (batch, time, feature)
        x = self.dense(x)  # (batch, time, feature)
        x = x.transpose(0, 1)  # (time, batch, feature)
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, hidden_size)
        return self.final_fc(x), (hn, cn)