import torch
import torch.nn as nn

class ClassicMLPNet(nn.Module):
    '''
    Classe que implementa uma rede neural clássica do tipo Multi-Layer Perceptron (MLP).
    Este modelo é usado como o baseline clássico para comparação de desempenho.
    '''
    def __init__(self, input_dim, hidden_size, dropout_rate, num_layers):
        '''
        Cria as camadas ocultas do modelo baseado no número de camadas e neurônios.
        '''
        super(ClassicMLPNet, self).__init__()
        layers = []
        current_dim = input_dim
        
        # Adiciona sequencialmente as camadas Lineares, Funções de Ativação e Dropout
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.SiLU())  
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_size
            
        # Adiciona a camada de saída única para a regressão
        layers.append(nn.Linear(current_dim, 1))
        
        # Combina todas as camadas em um objeto nn.Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        '''
        Define a propagação dos dados pela rede MLP clássica.
        '''
        return self.net(x).squeeze(-1)