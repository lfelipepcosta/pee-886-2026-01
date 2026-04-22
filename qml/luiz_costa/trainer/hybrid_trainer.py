import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from qml.luiz_costa.models.hybrid import HybridQuantumNet

class PyTorchHybridWrapper(BaseEstimator, RegressorMixin):
    '''
    Wrapper para tornar o modelo híbrido compatível com a API do Scikit-Learn.
    Permite usar o modelo em pipelines e funções de validação cruzada.
    '''
    def __init__(self, quantum_layers=1, n_qubits=4, hidden_size=64, learning_rate=1e-3, batch_size=256, epochs=50, patience=5, random_state=42, verbose=False):
        '''
        Inicializa os hiperparâmetros de treinamento e a arquitetura do modelo.
        '''
        self.quantum_layers = quantum_layers
        self.n_qubits = n_qubits
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        # Define o dispositivo como CPU para evitar incompatibilidades com o simulador quântico
        self.device = torch.device('cpu') 
        
    def fit(self, X, y):
        '''
        Realiza o treinamento do modelo híbrido utilizando PyTorch.
        '''
        # Fixa as sementes aleatórias para garantir reprodutibilidade
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Converte os dados para arrays NumPy se necessário
        X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
        y_arr = np.array(y)
        
        # Divide os dados em treino e validação interna para salvar o melhor modelo
        X_t, X_v, y_t, y_v = train_test_split(X_arr, y_arr, test_size=0.15, random_state=self.random_state)
        
        # Cria DataLoaders para iterar os dados em pacotes (batches)
        train_loader = DataLoader(TensorDataset(torch.tensor(X_t, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_v, dtype=torch.float32), torch.tensor(y_v, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)

        # Inicializa a rede neural híbrida
        self.input_dim = X_arr.shape[1]
        self.model = HybridQuantumNet(self.input_dim, quantum_layers=self.quantum_layers, n_qubits=self.n_qubits, hidden_size=self.hidden_size)
        
        # Inicializa o viés da última camada com a média do sinal para acelerar a convergência
        with torch.no_grad():
            nn.init.constant_(self.model.post_net.bias, y_arr.mean())
            
        self.model.to(self.device)
        
        # Define a função de perda (MSE) e o otimizador AdamW
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4) # AdamW com regularização weight decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        
        if self.verbose: 
            print(f"Iniciando treinamento híbrido no dispositivo {self.device}")

        # Loop de treinamento por épocas
        for epoch in range(self.epochs):
            self.model.train() # Coloca o modelo em modo de treinamento
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad() # Zera os gradientes acumulados
                outputs = self.model(batch_X) 
                loss = criterion(outputs, batch_y) 
                loss.backward() # Calcula os gradientes usando backpropagation
                optimizer.step() # Atualiza os pesos
            
            # Avalia o modelo no conjunto de validação interna
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss) # Reduz a taxa de aprendizagem se a perda estagnar
            
            # Verifica se houve melhora na perda para salvar o melhor estado
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Interrompe o treinamento se não houver melhora após 'patience' épocas
            if patience_counter >= self.patience:
                if self.verbose: 
                    print(f"Interrupção antecipada na época {epoch}. Melhor Validação: {best_val_loss:.4f}")
                break
                
        # Carrega os pesos da melhor versão encontrada durante as épocas
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        self.model.cpu() # Retorna o modelo para a CPU após o fim do treinamento
        return self

    def predict(self, X):
        '''
        Realiza a predição de sinal utilizando o modelo treinado.
        '''
        X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
        loader = DataLoader(TensorDataset(torch.tensor(X_arr, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)
        
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X in loader:
                # Executa a inferência nos pacotes de dados
                predictions.extend(self.model(batch_X[0].to(self.device)).cpu().numpy())
                
        self.model.cpu()
        return np.array(predictions)
        
    def __getstate__(self):
        '''
        Prepara a instância para ser serializada (via joblib/pickle).
        Garante que o modelo esteja na CPU para evitar erros com tensores CUDA.
        '''
        state = self.__dict__.copy()
        if self.model is not None:
            self.model.cpu()
        return state