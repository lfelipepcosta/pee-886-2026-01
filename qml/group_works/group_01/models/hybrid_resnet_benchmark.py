import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml

class QuantumCircuit(nn.Module):
    """
    Módulo PyTorch que encapsula o Circuito Quântico Variacional (VQC)
    Aceita o método de diferenciação como parâmetro para fins de benchmark.
    """
    def __init__(self, n_qubits=4, q_depth=2, diff_method="adjoint"):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.diff_method = diff_method
        
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Decorador configurado com o método repassado
        @qml.qnode(self.dev, interface="torch", diff_method=self.diff_method)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
        weight_shapes = {"weights": (q_depth, n_qubits, 3)}
        self.vqc = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        # O parameter-shift e finite-diff não suportam gradientes com batch (broadcasting) no PennyLane atual.
        # Portanto, iteramos sobre o batch manualmente nesses casos.
        if self.diff_method in ["parameter-shift", "finite-diff"]:
            return torch.stack([self.vqc(xi) for xi in x])
        return self.vqc(x)


class HybridResNet18Benchmark(nn.Module):
    """
    Modelo híbrido de classificação de imagens adaptado para benchmark 
    dos métodos de diferenciação quântica.
    """
    def __init__(self, num_classes=2, n_qubits=4, q_depth=2, diff_method="adjoint"):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, n_qubits)
        
        self.quantum_circuit = QuantumCircuit(n_qubits, q_depth, diff_method)
        self.fc_out = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        current_device = x.device
        
        # Processamento quântico (VQC) forçado na CPU para evitar overheads e conflitos de compilador (NVRTC)
        x_cpu = x.cpu()
        x_q = self.quantum_circuit(x_cpu)
        
        x_q = x_q.to(current_device)
        x_out = self.fc_out(x_q)
        return x_out
