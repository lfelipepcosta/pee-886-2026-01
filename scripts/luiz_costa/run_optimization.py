import os
import sys
import json
import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(repo_root)

from qml.luiz_costa.loaders.data_loader import DataLoader5G
from qml.luiz_costa.loaders.data_processor import create_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURE
from qml.luiz_costa.trainer.hybrid_trainer import PyTorchHybridWrapper

OUTPUT_DIR = os.path.join(repo_root, "data", "luiz_costa", "config")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PARAMS_FILE = os.path.join(OUTPUT_DIR, "best_params_hybrid.json")

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'quantum_layers': trial.suggest_int('quantum_layers', 1, 3), 
        'n_qubits': trial.suggest_categorical('n_qubits', [4]),
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]), 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]), 
        'epochs': 20, 
        'verbose': False
    }

    model = PyTorchHybridWrapper(**params)
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

def main():
    print("Otimização de Hiperparâmetros Híbridos")
    loader = DataLoader5G()
    df = loader.load_all_datasets()

    target = 'SS-RSRP'
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURE
    df = df.dropna(subset=features + [target])

    sample_size = min(30000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    X, y = df_sample[features], df_sample[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction="minimize", study_name="Hybrid_QML_Optimization")
    study.optimize(lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val), n_trials=15)
    
    with open(PARAMS_FILE, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("Otimização concluída com sucesso.")

if __name__ == "__main__":
    main()