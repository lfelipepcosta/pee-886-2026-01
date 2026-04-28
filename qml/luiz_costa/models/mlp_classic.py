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
        Implementa uma trava física projetada para evitar "Extrapolação Catastrófica"
        (Catastrophic Extrapolation) em dados Out-of-Distribution (OOD).
        '''
        out = self.net(x).squeeze(-1)
        
        # Justificativa Tanh:
        # O clamp rígido zera o gradiente (Dead Gradient) para valores iniciais fora do limite, impedindo o aprendizado pelo otimizador
        # O Tanh mapeia qualquer valor irreal de rede ([-inf, +inf]) suavemente para [-1, 1], mantendo a derivada contínua para o Backpropagation
        # A rede é multiplicada e deslocada para que sua saída limite coincida exatamente com as restrições teóricas de hardware e física:
        #   - Limite Inferior (-174 dBm): Piso termodinâmico de Ruído Térmico de Johnson-Nyquist
        #   - Limite Superior (-30 dBm): Ponto de saturação do LNA (Low Noise Amplifier) do receptor
        #   O limite superior não é restrito ao máximo empírico do dataset (-56 dBm)
        #   Isso permite que a rede aprenda a extrapolação física correta caso seja exposta a locais (OOD) mais próximos da antena do que os
        #   medidos no Drive Test
        #   Matemática: [-1, 1] * 72 = [-72, 72]. Deslocando: [-72, 72] - 102 = [-174, -30] dBm
        return torch.tanh(out) * 72.0 - 102.0