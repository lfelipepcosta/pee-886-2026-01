import os
import sys
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Adiciona a raiz do repositório ao path para permitir imports dos módulos QML
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(repo_root)

from qml.luiz_costa.loaders.data_loader import DataLoader5G
from qml.luiz_costa.loaders.data_preprocessor import create_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURE
from qml.luiz_costa.trainer.decision_tree_trainer import DecisionTreeWrapper
from qml.luiz_costa.evaluation.cross_validation import run_kfold_validation, run_gkfold_validation
from qml.luiz_costa.visualization.plotting import plot_feature_importance, plot_actual_vs_predicted, plot_error_distribution, plot_learning_curve

# Define locais para salvamento de modelos e resultados no data/luiz_costa
CONFIG_FILE = os.path.join(repo_root, "data", "luiz_costa", "config", "best_params_dt.json")
RESULTS_DIR = os.path.join(repo_root, "data", "luiz_costa", "results_dt")
MODEL_DIR = os.path.join(repo_root, "data", "luiz_costa", "trained_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    '''
    Hub de execução para o treinamento do baseline Decision Tree Regressor.
    Gera relatórios de validação cruzada e importância de features em PDF.
    '''
    print("Iniciando treinamento do baseline Decision Tree")
    
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

    # Instancia o Decision Tree com as configurações otimizadas
    model = DecisionTreeWrapper(
        max_depth=best_params.get('max_depth', None),
        min_samples_split=best_params.get('min_samples_split', 2),
        min_samples_leaf=best_params.get('min_samples_leaf', 1),
        verbose=True
    )
    
    # Cria o pipeline do Sklearn
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])

    # Validação Cruzada K-Fold Padrão (Avalia estabilidade estatística)
    run_kfold_validation(pipeline, X, y, model_name="Decision Tree", n_splits=5, output_dir=RESULTS_DIR)

    # Estima o erro de generalização através de K-Fold Espacial (GroupKFold por Antena)
    groups = df['Antena_Lat'].astype(str) + "_" + df['Antena_Lon'].astype(str)
    run_gkfold_validation(pipeline, X, y, groups=groups, model_name="Decision Tree", n_splits=5, output_dir=RESULTS_DIR)

    # Treina o modelo para avaliação de gráficos de dispersão e distribuição
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset total: {len(X)} amostras únicas.")
    print(f"Treinamento:   {len(X_train)} amostras.")
    print(f"Teste:         {len(X_test)} amostras.")
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calcula as métricas de desempenho final no conjunto de teste
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    num_antenas = len(df[['Antena_Lat', 'Antena_Lon']].drop_duplicates())
    
    metrics_report = (
        f"Relatório de Métricas Finais (Amostra de Teste) - Decision Tree\n"
        f"{'='*60}\n"
        f"Antenas Únicas no Dataset (Treino+Teste): {num_antenas}\n"
        f"Dataset Total (S/ Antenas Blind): {len(X)} amostras\n"
        f"Amostras de Treino: {len(X_train)}\n"
        f"Amostras de Teste Interno: {len(X_test)}\n"
        f"{'-'*60}\n"
        f"RMSE : {rmse:.4f} dB\n"
        f"MAE  : {mae:.4f} dB\n"
        f"R2   : {r2:.4f}\n"
        f"{'='*60}\n"
    )
    
    metrics_path = os.path.join(RESULTS_DIR, "dt_final_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_report)
    print(f"Métricas finais salvas em {metrics_path}")

    # Gera análises visuais salvas em PDF 300 DPI
    target_label = target
    plot_learning_curve(pipeline.named_steps['model'].history_, "Decision Tree", target_label, output_dir=RESULTS_DIR)
    plot_feature_importance(pipeline, X_test, y_test, target_label, "Decision Tree", output_dir=RESULTS_DIR)
    plot_actual_vs_predicted(y_test, y_pred, target_label, "Decision Tree", output_dir=RESULTS_DIR)
    plot_error_distribution(y_test, y_pred, target_label, "Decision Tree", output_dir=RESULTS_DIR)

    # Salva o pipeline para processos de inferência geoespacial
    saved_model_path = os.path.join(MODEL_DIR, "best_pipeline_dt.joblib")
    joblib.dump(pipeline, saved_model_path)
    print("Processo Decision Tree concluído com sucesso")

if __name__ == "__main__":
    main()
