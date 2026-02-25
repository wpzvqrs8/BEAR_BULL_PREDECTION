import torch
import torch.nn as nn
from typing import Optional

class TimeSeriesRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 1, rnn_type: str = 'LSTM', dropout: float = 0.2):
        super(TimeSeriesRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")
            
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        if self.rnn_type == 'LSTM':
            out, _ = self.rnn(x)
        else:
            out, _ = self.rnn(x)
            
        # Get output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out) # Since our primary target is binary
        return out

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
