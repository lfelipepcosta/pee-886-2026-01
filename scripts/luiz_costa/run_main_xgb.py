import os
import sys
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Adiciona a raiz do repositório ao path para permitir imports dos módulos QML
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(repo_root)

from qml.luiz_costa.loaders.data_loader import DataLoader5G
from qml.luiz_costa.loaders.data_preprocessor import create_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURE
from qml.luiz_costa.trainer.xgboost_trainer import XGBoostWrapper
from qml.luiz_costa.evaluation.cross_validation import run_kfold_validation
from qml.luiz_costa.visualization.plotting import plot_feature_importance, plot_actual_vs_predicted, plot_error_distribution

# Define locais para salvamento de modelos e resultados no data/luiz_costa
CONFIG_FILE = os.path.join(repo_root, "data", "luiz_costa", "config", "best_params_xgb.json")
RESULTS_DIR = os.path.join(repo_root, "data", "luiz_costa", "results_xgb")
MODEL_DIR = os.path.join(repo_root, "data", "luiz_costa", "trained_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    '''
    Hub de execução para o treinamento do baseline XGBoost Regressor.
    Gera relatórios de validação cruzada e importância de features em PDF.
    '''
    print("Iniciando treinamento do baseline XGBoost")
    
    # Carrega parâmetros de busca do arquivo JSON, se disponíveis
    best_params = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            best_params = json.load(f)

    # Carrega os conjuntos de dados processados
    loader = DataLoader5G()
    df = loader.load_all_datasets()

    target = 'SS-RSRP'
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURE
    # Limpa linhas vazias nas features e alvo
    df = df.dropna(subset=features + [target])
    
    X, y = df[features], df[target]

    # Instancia o XGBoost com as configurações otimizadas
    model = XGBoostWrapper(
        n_estimators=best_params.get('n_estimators', 200),
        max_depth=best_params.get('max_depth', 6),
        learning_rate=best_params.get('learning_rate', 0.05),
        subsample=best_params.get('subsample', 1.0),
        colsample_bytree=best_params.get('colsample_bytree', 1.0),
        min_child_weight=best_params.get('min_child_weight', 1),
        verbose=True
    )
    
    # Cria o pipeline do Sklearn
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])

    # Realiza a avaliação estatística com K-Fold Cross Validation
    run_kfold_validation(pipeline, X, y, n_splits=5, output_dir=RESULTS_DIR)

    # Treina o modelo para avaliação de gráficos de dispersão e distribuição
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Gera análises visuais salvas em PDF 300 DPI
    target_label = f"{target}_XGB"
    plot_feature_importance(pipeline, X_test, y_test, target_label, output_dir=RESULTS_DIR)
    plot_actual_vs_predicted(y_test, y_pred, target_label, output_dir=RESULTS_DIR)
    plot_error_distribution(y_test, y_pred, target_label, output_dir=RESULTS_DIR)

    # Salva o pipeline para processos de inferência geoespacial
    saved_model_path = os.path.join(MODEL_DIR, "best_pipeline_xgb.joblib")
    joblib.dump(pipeline, saved_model_path)
    print("Processo XGBoost concluído com sucesso")

if __name__ == "__main__":
    main()