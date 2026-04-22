import torch
import torch.nn as nn
import pennylane as qml

class HybridQuantumNet(nn.Module):
    def __init__(self, input_dim, quantum_layers=1, n_qubits=4, hidden_size=64):
        super(HybridQuantumNet, self).__init__()
        self.n_qubits = n_qubits
        
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),          
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.n_qubits), 
            nn.Tanh() 
        )
        
        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
        weight_shapes = {"weights": (quantum_layers, self.n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        self.post_net = nn.Linear(self.n_qubits, 1)

    def forward(self, x):
        x_classical = self.pre_net(x)
        x_quantum = self.quantum_layer(x_classical)
        out = self.post_net(x_quantum).squeeze(-1)
        return out