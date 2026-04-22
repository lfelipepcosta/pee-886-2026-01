import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

class XGBoostWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05, 
                 subsample=1.0, colsample_bytree=1.0, min_child_weight=1, 
                 random_state=42, verbose=False):
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
        X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
        y_arr = np.array(y)
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            objective='reg:squarederror',
            tree_method='hist',
            device='cuda',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        if self.verbose:
            print("Treinando baseline clássico XGBoost...")
            
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("O modelo ainda não foi treinado (fitted).")
            
        X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
        return self.model.predict(X_arr)