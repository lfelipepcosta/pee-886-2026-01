import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

# Adiciona a raiz do repositório ao path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(repo_root)

from qml.luiz_costa.loaders.data_loader import DataLoader5G
from qml.luiz_costa.loaders.data_preprocessor import create_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURE
from qml.luiz_costa.trainer.xgboost_trainer import XGBoostWrapper
from qml.luiz_costa.trainer.hybrid_trainer import PyTorchHybridWrapper
from sklearn.neural_network import MLPRegressor

RESULTS_DIR = os.path.join(repo_root, "data", "luiz_costa", "results_generalization")
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    return rmse, mae

def main():
    print("Iniciando Teste de Generalização (Leave-Antenna-Out)")
    
    loader = DataLoader5G()
    df = loader.load_all_datasets()
    
    target = 'SS-RSRP'
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURE
    df = df.dropna(subset=features + [target])
    
    # Cria identificadores únicos para cada Antena
    df['Antenna_ID'] = df['Antena_Lat'].astype(str) + "_" + df['Antena_Lon'].astype(str)
    
    X, y, groups = df[features], df[target], df['Antenna_ID']
    print(f"Dataset total: {len(X)} amostras")
    print(f"Número de Antenas Únicas (Grupos): {groups.nunique()}")
    
    # Função para carregar os hiperparâmetros salvos pelo Optuna
    def load_params(filename):
        filepath = os.path.join(repo_root, "data", "luiz_costa", "config", filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return {}

    p_xgb = load_params("best_params_xgb.json")
    p_mlp = load_params("best_params_mlp.json")
    p_hyb = load_params("best_params_hybrid.json")

    # Configura os 3 modelos com os hiperparâmetros otimizados (ou os mantêm como fallback)
    models = {
        "XGBoost": XGBoostWrapper(
            n_estimators=p_xgb.get('n_estimators', 200),
            max_depth=p_xgb.get('max_depth', 6),
            learning_rate=p_xgb.get('learning_rate', 0.05),
            subsample=p_xgb.get('subsample', 1.0),
            colsample_bytree=p_xgb.get('colsample_bytree', 1.0),
            min_child_weight=p_xgb.get('min_child_weight', 1),
            verbose=False
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=p_mlp.get('hidden_layer_sizes', (128, 64)),
            activation=p_mlp.get('activation', 'relu'),
            alpha=p_mlp.get('alpha', 0.0001),
            learning_rate_init=p_mlp.get('learning_rate_init', 0.001),
            max_iter=50, 
            random_state=42
        ),
        "Hybrid": PyTorchHybridWrapper(
            quantum_layers=p_hyb.get('quantum_layers', 2),
            n_qubits=p_hyb.get('n_qubits', 4),
            hidden_size=p_hyb.get('hidden_size', 128),
            learning_rate=p_hyb.get('learning_rate', 0.005),
            batch_size=p_hyb.get('batch_size', 256),
            epochs=15,  # 15 épocas são suficientes para K-Fold rápido
            verbose=False
        )
    }
    
    results = {name: {'rmse': [], 'mae': []} for name in models}
    
    # O GroupKFold garante que os dados de uma Antena NUNCA estejam no treino e teste ao mesmo tempo
    gkf = GroupKFold(n_splits=5)
    
    fold = 1
    for train_idx, test_idx in gkf.split(X, y, groups):
        print(f"\n--- Processando Fold Espacial {fold}/5 ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"Antenas no Treino: {groups.iloc[train_idx].nunique()} | Antenas no Teste: {groups.iloc[test_idx].nunique()}")
        
        for name, model in models.items():
            print(f"Treinando e avaliando {name}")
            rmse, mae = evaluate_model(model, X_train, y_train, X_test, y_test)
            results[name]['rmse'].append(rmse)
            results[name]['mae'].append(mae)
            print(f"  {name} -> RMSE: {rmse:.2f} dB")
            
        fold += 1
        
    print("\n" + "="*50)
    print("RELATÓRIO FINAL DE GENERALIZAÇÃO (Leave-Antenna-Out)")
    print("="*50)
    report = "Relatório de Generalização Espacial\n" + "="*40 + "\n"
    
    for name in models:
        mean_rmse = np.mean(results[name]['rmse'])
        std_rmse = np.std(results[name]['rmse'])
        mean_mae = np.mean(results[name]['mae'])
        
        res_str = f"{name}:\n  RMSE Médio: {mean_rmse:.2f} dB ± {std_rmse:.2f}\n  MAE Médio:  {mean_mae:.2f} dB\n"
        print(res_str)
        report += res_str + "\n"
        
    with open(os.path.join(RESULTS_DIR, "generalization_metrics.txt"), "w") as f:
        f.write(report)
        
    print(f"Resultados salvos em {RESULTS_DIR}/generalization_metrics.txt")

if __name__ == "__main__":
    main()
