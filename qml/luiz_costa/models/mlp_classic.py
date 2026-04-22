import torch
import torch.nn as nn

class ClassicMLPNet(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout_rate, num_layers):
        super(ClassicMLPNet, self).__init__()
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.SiLU())  
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_size
            
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)