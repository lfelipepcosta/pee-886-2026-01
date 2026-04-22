import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from qml.luiz_costa.models.mlp_classic import ClassicMLPNet

class PyTorchMLPWrapper(BaseEstimator, RegressorMixin):
    '''
    Classe que envolve o modelo MLP para torná-lo compatível com o Scikit-Learn.
    Facilita o treinamento, avaliação e uso em pipelines de regressão.
    '''
    def __init__(self, hidden_size=128, dropout_rate=0.1, num_layers=3, 
                 learning_rate=1e-3, batch_size=512, epochs=200, patience=15, 
                 random_state=42, verbose=False):
        '''
        Define os parâmetros do otimizador e a estrutura da rede neural.
        '''
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        # Utiliza GPU (CUDA) se disponível, caso contrário usa a CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        '''
        Treina o modelo MLP clássico utilizando o conjunto de dados.
        '''
        # Define sementes aleatórias para reprodutibilidade no PyTorch e NumPy
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)
        
        # Converte entradas para tensores Float32
        X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
        y_arr = np.array(y)
        
        # Cria um subconjunto de validação para monitorar o overfit durante o treino
        X_t, X_v, y_t, y_v = train_test_split(X_arr, y_arr, test_size=0.15, random_state=self.random_state)
        
        # Prepara os DataLoaders para carregamento eficiente durante as épocas
        train_loader = DataLoader(TensorDataset(torch.tensor(X_t, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_v, dtype=torch.float32), torch.tensor(y_v, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)

        # Instancia a arquitetura da rede MLP
        input_dim = X_arr.shape[1]
        self.model = ClassicMLPNet(input_dim, self.hidden_size, self.dropout_rate, self.num_layers)
        self.model.to(self.device)
        
        # Define o critério de erro (MSE) e o otimizador AdamW com regularização
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        
        if self.verbose:
            print(f"Treinando baseline MLP clássico no dispositivo {self.device}")

        # Loop principal de treinamento
        for epoch in range(self.epochs):
            self.model.train() # Habilita o modo de treinamento da rede
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Desabilita o treino para avaliar a métrica de erro no conjunto de validação
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            
            # Salva o estado atual se for o melhor resultado de validação até agora
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Interrompe o treino se não houver melhoria dentro do limite de paciência (Early Stopping)
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Interrupção antecipada na época {epoch}. Melhor perda na validação: {best_val_loss:.4f}")
                break
                
        # Mantém a rede com os melhores pesos encontrados antes da interrupção
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        self.model.cpu() 
        return self

    def predict(self, X):
        '''
        Gera predições de RSRP utilizando a rede MLP treinada.
        '''
        X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
        loader = DataLoader(TensorDataset(torch.tensor(X_arr, dtype=torch.float32)), batch_size=self.batch_size, shuffle=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Acumula as predições de cada batch
        predictions = []
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X[0].to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())
                
        self.model.cpu()
        return np.array(predictions)
    
    def __getstate__(self):
        '''
        Prepara o estado do objeto para salvamento limpo, evitando bugs de memória em processos GPU.
        '''
        state = self.__dict__.copy()
        if self.model is not None:
            self.model.cpu() 
        return state