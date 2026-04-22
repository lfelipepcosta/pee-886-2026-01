import torch
import torch.nn as nn
import pennylane as qml

class HybridQuantumNet(nn.Module):
    '''
    Classe que implementa a rede neural híbrida quântico-clássica.
    Combina camadas densas tradicionais com um circuito quântico variacional.
    '''
    def __init__(self, input_dim, quantum_layers=1, n_qubits=4, hidden_size=64):
        '''
        Inicializa as redes clássicas de pré e pós-processamento e o QNode quântico.
        '''
        super(HybridQuantumNet, self).__init__()
        self.n_qubits = n_qubits
        
        # Rede clássica para preparar as features para o circuito quântico
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),          
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.n_qubits), 
            nn.Tanh() # Converte a saída para o intervalo [-1, 1], adequado para AngleEmbedding
        )
        
        # Cria o dispositivo PennyLane para simular os qubits
        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            '''
            Define o circuito quântico com AngleEmbedding e camadas de emaranhamento.
            '''
            # Codifica os dados clássicos nos ângulos de rotação do estado quântico
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # Aplica camadas parametrizadas de portas quânticas fortemente emaranhadas
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # Realiza a medição do valor esperado de Pauli-Z para cada qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
        # Envolve o QNode em uma camada TorchLayer para integração direta com o PyTorch
        weight_shapes = {"weights": (quantum_layers, self.n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Camada densa final para converter a saída quântica na predição final de RSRP
        self.post_net = nn.Linear(self.n_qubits, 1)

    def forward(self, x):
        '''
        Define o fluxo de dados (forward pass) do modelo híbrido.
        '''
        # Passa pela rede clássica inicial
        x_classical = self.pre_net(x)
        
        # Processa os dados no circuito quântico
        x_quantum = self.quantum_layer(x_classical)
        
        # Finaliza com a última camada densa e remove a dimensão extra
        out = self.post_net(x_quantum).squeeze(-1)
        return out