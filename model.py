# model.py
# autrhor: Rithwik Nambiar
# Date: 2025_09_17


import torch
import torch.nn as nn


class ProteinClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)
    
def load_trained_model(model_path, input_dim, hidden_dim, output_dim, device ="cpu"):
    
    model = ProteinClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model