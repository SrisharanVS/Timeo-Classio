import torch
import torch.nn as nn

class MAPEPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size=64):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4)  # 4 outputs for different models
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.batch_norm2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.batch_norm3(x)
        x = self.fc3(x)
        return x 