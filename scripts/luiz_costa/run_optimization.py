import os
import sys
import json
import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Adiciona a raiz do repositório ao path para permitir imports dos módulos QML
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(repo_root)

from qml.luiz_costa.loaders.data_loader import DataLoader5G
from qml.luiz_costa.loaders.data_preprocessor import create_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURE
from qml.luiz_costa.trainer.hybrid_trainer import PyTorchHybridWrapper

# Diretórios para salvar os melhores parâmetros encontrados pelo Optuna
OUTPUT_DIR = os.path.join(repo_root, "data", "luiz_costa", "config")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PARAMS_FILE = os.path.join(OUTPUT_DIR, "best_params_hybrid.json")

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    '''
    Função objetivo do Optuna para minimizar o RMSE na validação hibrida.
    '''
    # Define o espaço de busca dos hiperparâmetros quânticos e clássicos
    params = {
        'quantum_layers': trial.suggest_int('quantum_layers', 1, 3), 
        'n_qubits': trial.suggest_categorical('n_qubits', [4]), # Qubits fixados em 4 para evitar barren plateaus
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]), 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]), 
        'epochs': 20, # Treino curto apenas para triagem de parâmetros
        'verbose': False 
    }

    # Inicializa o Wrapper híbrido e treina com as sugestões do Optuna
    model = PyTorchHybridWrapper(**params)
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    # Retorna o RMSE como métrica de otimização
    return np.sqrt(mean_squared_error(y_val, preds))

def main():
    '''
    Rotina de otimização automática de hiperparâmetros para o modelo Quântico Híbrido.
    '''
    print("Iniciando otimização TPE para modelos híbridos")
    loader = DataLoader5G()
    df = loader.load_all_datasets()

    target = 'SS-RSRP'
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURE
    df = df.dropna(subset=features + [target])

    # Amostra um conjunto menor (30k) para agilizar o processo variacional quântico
    sample_size = min(30000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    X, y = df_sample[features], df_sample[target]

    # Divide os dados amostrados para o processo iterativo do Optuna
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria o estudo do Optuna e inicia a otimização de 15 tentativas (trials)
    study = optuna.create_study(direction="minimize", study_name="Hybrid_QML_Optimization")
    study.optimize(lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val), n_trials=15)
    
    # Exporta o dicionário de melhores parâmetros para um arquivo JSON
    with open(PARAMS_FILE, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("Otimização híbrida concluída")

if __name__ == "__main__":
    main()