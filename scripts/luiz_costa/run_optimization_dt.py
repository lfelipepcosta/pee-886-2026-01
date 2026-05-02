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
from qml.luiz_costa.trainer.decision_tree_trainer import DecisionTreeWrapper

# Local para salvar o melhor conjunto de hiperparâmetros encontrado
OUTPUT_DIR = os.path.join(repo_root, "data", "luiz_costa", "config")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PARAMS_FILE = os.path.join(OUTPUT_DIR, "best_params_dt.json")

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    '''
    Função objetivo para minimizar o RMSE do Decision Tree através do Optuna.
    '''
    # Espaço de busca focado em FORTE regularização para combater Overfitting
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_samples_split': trial.suggest_int('min_samples_split', 20, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
        'random_state': 42,
        'verbose': False 
    }

    # Inicializa modelo e pipeline
    model = DecisionTreeWrapper(**params)
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])

    # Executa o ajuste dos pesos e retorna o erro quadrático médio (RMSE)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

def main():
    '''
    Inicia o processo de busca automática de hiperparâmetros para a Árvore de Decisão.
    '''
    print("Iniciando otimização paramétrica Decision Tree clássico")
    loader = DataLoader5G()
    df = loader.load_all_datasets()

    target = 'SS-RSRP'
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURE
    df = df.dropna(subset=features + [target])

    X, y = df[features], df[target]
    # Divisão dos dados para validação interna do processo de HPO
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # Executa o estudo Optuna com 30 tentativas diferentes
    study = optuna.create_study(direction="minimize", study_name="DT_5G_Optimization")
    study.optimize(lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val), n_trials=30)
    
    # Salva o arquivo de configuração JSON com os melhores resultados
    with open(PARAMS_FILE, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Otimização Decision Tree finalizada. Parâmetros em {PARAMS_FILE}")

if __name__ == "__main__":
    main()
