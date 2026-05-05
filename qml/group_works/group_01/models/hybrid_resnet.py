import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml

class QuantumCircuit(nn.Module):
    """
    Módulo PyTorch que encapsula o Circuito Quântico Variacional (VQC)
    Utiliza PennyLane para definir e executar operações quânticas
    """
    def __init__(self, n_qubits=4, q_depth=2):
        """
        Inicializa o circuito quântico
        
        Args:
            n_qubits (int): Número de qubits a serem utilizados.
            q_depth (int): Profundidade (número de camadas) do Ansatz quântico.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # diff_method="adjoint": É o método de cálculo de gradientes quânticos (Adjoint Differentiation)
        # É matematicamente equivalente ao Backpropagation clássico, porém adaptado para circuitos quânticos
        # Ele é muito mais rápido e consome menos memória do que o método tradicional "parameter-shift"
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def qnode(inputs, weights):
            """
            Define as portas e operações do circuito quântico.
            """
            # Angle Encoding
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # StronglyEntanglingLayers processa a informação no espaço de Hilbert
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # Medição
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
        # Formato dos pesos treináveis do circuito
        weight_shapes = {"weights": (q_depth, n_qubits, 3)}
        
        # Converte o QNode em uma camada PyTorch (TorchLayer) para uso no modelo híbrido
        self.vqc = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        """
        Passo forward da camada quântica
        """
        return self.vqc(x)


class HybridResNet18(nn.Module):
    """
    Modelo híbrido de classificação de imagens (Clássico-Quântico).
    Utiliza a ResNet-18 para extração clássica de características e um 
    Circuito Quântico Variacional (VQC) para o processamento final.
    """
    def __init__(self, num_classes=2, n_qubits=4, q_depth=2):
        """
        Inicializa a arquitetura híbrida.
        
        Args:
            num_classes (int): Número de classes alvo (2 para benigno/maligno).
            n_qubits (int): Número de qubits e saídas do dressing clássico.
            q_depth (int): Profundidade do circuito quântico.
        """
        super().__init__()
        # Carrega a rede ResNet-18 pré-treinada (ImageNet)
        self.resnet = models.resnet18(pretrained=True)
        
        # Congela todas as camadas convolucionais
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Substitui a última camada totalmente conectada (Dressing Clássico)
        num_ftrs = self.resnet.fc.in_features  # Originalmente 512 atributos para a ResNet-18
        
        # Camada linear para reduzir de 512 para 4 valores reais (para entrar no circuito de 4 qubits)
        self.resnet.fc = nn.Linear(num_ftrs, n_qubits)
        
        # Camada de Circuito Quântico (VQC)
        self.quantum_circuit = QuantumCircuit(n_qubits, q_depth)
        
        # Camada de Saída clássica (pós-processamento das medições quânticas)
        self.fc_out = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        """
        Passo forward de todo o modelo híbrido
        """
        # Extração de características e redução clássica (Dressing)
        x = self.resnet(x)
        
        # Salva o dispositivo atual (GPU se disponível) e move para a CPU
        # Isso evita erros do compilador dinâmico da Nvidia (NVRTC) no VQC
        # Além disso, para 4 qubits, rodar na CPU é muito mais eficiente.
        current_device = x.device
        x_cpu = x.cpu()
        
        # Processamento quântico (VQC) na CPU
        x_q = self.quantum_circuit(x_cpu)
        
        # Retorna o tensor para a GPU para a camada final
        x_q = x_q.to(current_device)
        
        # Saída final (classificação)
        x_out = self.fc_out(x_q)
        return x_out
