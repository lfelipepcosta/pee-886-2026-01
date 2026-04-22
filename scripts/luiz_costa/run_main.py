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
from qml.luiz_costa.trainer.hybrid_trainer import PyTorchHybridWrapper
from qml.luiz_costa.evaluation.cross_validation import run_kfold_validation
from qml.luiz_costa.visualization.plotting import plot_feature_importance, plot_actual_vs_predicted, plot_error_distribution

# Define caminhos para arquivos de configuração e diretórios de resultados
CONFIG_FILE = os.path.join(repo_root, "data", "luiz_costa", "config", "best_params_hybrid.json")
RESULTS_DIR = os.path.join(repo_root, "data", "luiz_costa", "results_hybrid")
MODEL_DIR = os.path.join(repo_root, "data", "luiz_costa", "trained_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    '''
    Orquestra o fluxo de treinamento completo para o modelo Híbrido Quântico.
    Carrega dados, treina, avalia métricas e gera gráficos estatísticos.
    '''
    print("Iniciando treinamento da rede híbrida quântica")
    
    # Carrega os melhores parâmetros encontrados via Optuna, se existirem
    best_params = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            best_params = json.load(f)

    # Inicializa o carregamento dos dados consolidados (cache)
    loader = DataLoader5G()
    df = loader.load_all_datasets()

    target = 'SS-RSRP'
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURE
    # Limpa linhas com valores nulos nas features selecionadas
    df = df.dropna(subset=features + [target])
    
    X, y = df[features], df[target]

    # Configura o Wrapper do PyTorch com os hiperparâmetros carregados
    model = PyTorchHybridWrapper(
        quantum_layers=best_params.get('quantum_layers', 2),
        n_qubits=best_params.get('n_qubits', 4),
        hidden_size=best_params.get('hidden_size', 128),
        learning_rate=best_params.get('learning_rate', 0.005),
        batch_size=best_params.get('batch_size', 256),
        epochs=50, 
        verbose=True
    )
    
    # Cria a pipeline integrando pré-processador e o modelo quântico
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])

    # Executa a validação cruzada para estimar a flutuação estatística do erro
    run_kfold_validation(pipeline, X, y, n_splits=5, output_dir=RESULTS_DIR)

    # Treina o modelo final com 80% dos dados e avalia nos 20% restantes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Gera e salva os gráficos analíticos em formato PDF
    target_label = f"{target}_Hybrid"
    plot_feature_importance(pipeline, X_test, y_test, target_label, output_dir=RESULTS_DIR)
    plot_actual_vs_predicted(y_test, y_pred, target_label, output_dir=RESULTS_DIR)
    plot_error_distribution(y_test, y_pred, target_label, output_dir=RESULTS_DIR)
    
    # Salva o pipeline treinado em disco para uso posterior em inferência
    saved_model_path = os.path.join(MODEL_DIR, "best_pipeline_hybrid.joblib")
    joblib.dump(pipeline, saved_model_path)
    print("Processo híbrido concluído com sucesso")

if __name__ == "__main__":
    main()