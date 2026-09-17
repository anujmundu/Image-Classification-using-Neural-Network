import torch
import torch.nn as nn

def build_rnn(num_classes: int = 10, input_size: int = 3, hidden_size: int = 64, num_layers: int = 1, seq_len: int = 32, **kwargs) -> nn.Module:
    """Simple RNN backbone for sequential image data.
    Expects input of shape (batch, seq_len, input_size).
    """
    class RNNBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            # x: (B, L, C)
            out, _ = self.rnn(x)
            # take last time step
            out = out[:, -1, :]
            return self.fc(out)
    return RNNBackbone()


def build_lstm(num_classes: int = 10, input_size: int = 3, hidden_size: int = 64, num_layers: int = 1, seq_len: int = 32, **kwargs) -> nn.Module:
    """Simple LSTM backbone for sequential image data.
    Expects input of shape (batch, seq_len, input_size).
    """
    class LSTMBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)
    return LSTMBackbone()
