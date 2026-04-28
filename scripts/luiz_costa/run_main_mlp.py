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
from qml.luiz_costa.trainer.mlp_trainer import PyTorchMLPWrapper
from qml.luiz_costa.evaluation.cross_validation import run_kfold_validation, run_gkfold_validation
from qml.luiz_costa.visualization.plotting import plot_feature_importance, plot_actual_vs_predicted, plot_error_distribution, plot_learning_curve

# Configura diretórios de salvamento e leitura de parâmetros JSON
CONFIG_FILE = os.path.join(repo_root, "data", "luiz_costa", "config", "best_params_mlp.json")
RESULTS_DIR = os.path.join(repo_root, "data", "luiz_costa", "results_mlp")
MODEL_DIR = os.path.join(repo_root, "data", "luiz_costa", "trained_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():     
    '''
    Executa o treinamento do baseline clássico MLP Perceptron.
    Gera métricas de validação cruzada e gráficos em PDF.
    '''
    print("Iniciando treinamento do baseline clássico MLP")
    
    # Carrega hiperparâmetros salvos previamente
    best_params = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            best_params = json.load(f)

    # Carrega dados processados 5G
    loader = DataLoader5G()
    df = loader.load_all_datasets()

    target = 'SS-RSRP'
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURE
    # Filtra dados faltantes
    df = df.dropna(subset=features + [target])
    
    X, y = df[features], df[target]

    # Instancia o wrapper para treinamento em PyTorch (CPU ou GPU)
    model = PyTorchMLPWrapper(
        hidden_size=best_params.get('hidden_size', 256),
        num_layers=best_params.get('num_layers', 3),
        dropout_rate=best_params.get('dropout_rate', 0.1),
        learning_rate=best_params.get('learning_rate', 0.001),
        batch_size=best_params.get('batch_size', 512),
        epochs=50, 
        patience=7,
        verbose=True
    )
    
    # Monta a pipeline clássica
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])

    # Estima o erro médio através de K-Fold Validation
    run_kfold_validation(pipeline, X, y, n_splits=5, output_dir=RESULTS_DIR)

    # Estima o erro de generalização através de K-Fold Espacial (GroupKFold por Antena)
    groups = df['Antena_Lat'].astype(str) + "_" + df['Antena_Lon'].astype(str)
    run_gkfold_validation(pipeline, X, y, groups=groups, n_splits=5, output_dir=RESULTS_DIR)

    # Treina o modelo final e gera as predições de teste
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
    
    metrics_report = (
        f"Relatório de Métricas Finais (Amostra de Teste) - MLP Clássico\n"
        f"{'='*60}\n"
        f"RMSE : {rmse:.4f} dB\n"
        f"MAE  : {mae:.4f} dB\n"
        f"R2   : {r2:.4f}\n"
        f"{'='*60}\n"
    )
    
    metrics_path = os.path.join(RESULTS_DIR, "mlp_final_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_report)
    print(f"Métricas finais salvas em {metrics_path}")

    # Renderiza e salva resultados visuais em PDF
    target_label = f"{target}_MLP"
    plot_learning_curve(pipeline.named_steps['model'].history_, "MLP", target_label, output_dir=RESULTS_DIR)
    plot_feature_importance(pipeline, X_test, y_test, target_label, output_dir=RESULTS_DIR)
    plot_actual_vs_predicted(y_test, y_pred, target_label, output_dir=RESULTS_DIR)
    plot_error_distribution(y_test, y_pred, target_label, output_dir=RESULTS_DIR)

    # Salva o pipeline via joblib para uso em inferência futura
    saved_model_path = os.path.join(MODEL_DIR, "best_pipeline_mlp.joblib")
    joblib.dump(pipeline, saved_model_path)
    print("Processo MLP concluído com sucesso")

if __name__ == "__main__":
    main()