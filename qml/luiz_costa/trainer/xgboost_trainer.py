import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin

class XGBoostWrapper(BaseEstimator, RegressorMixin):
    '''
    Envoltório compatível com Scikit-Learn para o modelo XGBoost Regressor.
    Permite integrar o XGBoost em pipelines de processamento e validação.
    '''
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05, 
                 subsample=1.0, colsample_bytree=1.0, min_child_weight=1, 
                 random_state=42, verbose=False):
        '''
        Configura os hiperparâmetros principais do ensemble de árvores de decisão.
        '''
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None

    def fit(self, X, y):
        '''
        Treina o modelo XGBoost utilizando os dados fornecidos.
        Acelera o treinamento via hardware CUDA se configurado.
        '''
        # Transforma os dados em arrays compatíveis para o treinamento
        X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
        y_arr = np.array(y)
        
        # Inicializa o regressor XGBoost com parâmetros de hardware específicos
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            objective='reg:squarederror',
            tree_method='hist',
            device='cuda', # Tenta usar GPU para acelerar o baseline clássico
            random_state=self.random_state,
            n_jobs=-1
        )
        
        if self.verbose:
            print("Treinando baseline clássico acelerado XGBoost")
            
        # Realiza o ajuste dos pesos do modelo com conjunto de avaliação para curva de aprendizado
        X_t, X_v, y_t, y_v = train_test_split(X_arr, y_arr, test_size=0.15, random_state=self.random_state)
        
        self.model.fit(
            X_t, y_t,
            eval_set=[(X_t, y_t), (X_v, y_v)],
            verbose=False
        )
        
        # Armazena o histórico de métricas
        results = self.model.evals_result()
        self.history_ = {
            'train_loss': results['validation_0']['rmse'],
            'val_loss': results['validation_1']['rmse']
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
        # Retorna as predições geradas pela floresta de decisão
        return self.model.predict(X_arr)