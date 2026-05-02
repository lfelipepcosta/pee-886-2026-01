import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class DecisionTreeWrapper(BaseEstimator, RegressorMixin):
    '''
    Envoltório compatível com Scikit-Learn para o modelo Decision Tree Regressor.
    Permite integrar a Árvore de Decisão em pipelines de processamento e validação.
    '''
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, random_state=42, verbose=False):
        '''
        Configura os hiperparâmetros principais da árvore de decisão.
        '''
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None

    def fit(self, X, y):
        '''
        Treina o modelo Decision Tree.
        '''
        # Transforma os dados em arrays compatíveis para o treinamento
        X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
        y_arr = np.array(y)
        
        # Inicializa o regressor Decision Tree com parâmetros
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        if self.verbose:
            print("Treinando Decision Tree")
            
        # Realiza a divisão com conjunto de avaliação para simular a curva de aprendizado
        X_t, X_v, y_t, y_v = train_test_split(X_arr, y_arr, test_size=0.15, random_state=self.random_state)
        
        self.model.fit(X_t, y_t)
        
        # Avalia a árvore no conjunto de treino e validação
        y_t_pred = self.model.predict(X_t)
        y_v_pred = self.model.predict(X_v)
        
        # O histórico de métricas é armazenado para compatibilidade com ferramentas visuais
        # Salva como MSE, pois a função plotting.py fará np.sqrt para modelos não-XGBoost
        mse_train = mean_squared_error(y_t, y_t_pred)
        mse_val = mean_squared_error(y_v, y_v_pred)
        
        self.history_ = {
            'train_loss': [mse_train, mse_train],
            'val_loss': [mse_val, mse_val]
        }
        return self

    def predict(self, X):
        '''
        Realiza a inferência de novos dados e retorna as estimativas de RSRP.
        '''
        # Verifica se o modelo já passou pelo processo de treinamento
        if self.model is None:
            raise ValueError("O modelo ainda não foi treinado (fitted)")
            
        X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
        # Retorna as predições geradas pela árvore de decisão
        return self.model.predict(X_arr)
