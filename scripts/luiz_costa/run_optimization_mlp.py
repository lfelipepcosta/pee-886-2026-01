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
from qml.luiz_costa.trainer.mlp_trainer import PyTorchMLPWrapper

OUTPUT_DIR = os.path.join(repo_root, "data", "luiz_costa", "config")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PARAMS_FILE = os.path.join(OUTPUT_DIR, "best_params_mlp.json")

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
        'num_layers': trial.suggest_int('num_layers', 2, 5),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.4),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]),
        'epochs': 150, 
        'random_state': 42,
        'verbose': False
    }

    model = PyTorchMLPWrapper(**params)
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])

    sample_size = min(500000, len(X_train))
    X_train_sub = X_train.sample(n=sample_size, random_state=42)
    y_train_sub = y_train.loc[X_train_sub.index]

    pipeline.fit(X_train_sub, y_train_sub)
    preds = pipeline.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

def main():
    print("Otimização de Hiperparâmetros (MLP Clássico)")
    loader = DataLoader5G()
    df = loader.load_all_datasets()

    target = 'SS-RSRP'
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURE
    df = df.dropna(subset=features + [target])

    X, y = df[features], df[target]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    study = optuna.create_study(direction="minimize", study_name="MLP_5G_Optimization")
    study.optimize(lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    
    with open(PARAMS_FILE, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Otimização concluída. Parâmetros salvos em: {PARAMS_FILE}")

if __name__ == "__main__":
    main()