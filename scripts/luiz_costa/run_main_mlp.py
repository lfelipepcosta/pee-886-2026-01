import os
import sys
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(repo_root)

from qml.luiz_costa.loaders.data_loader import DataLoader5G
from qml.luiz_costa.loaders.data_processor import create_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURE
from qml.luiz_costa.trainer.mlp_trainer import PyTorchMLPWrapper
from qml.luiz_costa.evaluation.cross_validation import run_kfold_validation
from qml.luiz_costa.visualization.plotting import plot_feature_importance, plot_actual_vs_predicted, plot_error_distribution

CONFIG_FILE = os.path.join(repo_root, "data", "luiz_costa", "config", "best_params_mlp.json")
RESULTS_DIR = os.path.join(repo_root, "data", "luiz_costa", "results_mlp")
MODEL_DIR = os.path.join(repo_root, "data", "luiz_costa", "trained_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():     
    print("Treinamento Baseline MLP")
    
    best_params = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            best_params = json.load(f)

    loader = DataLoader5G()
    df = loader.load_all_datasets()

    target = 'SS-RSRP'
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURE
    df = df.dropna(subset=features + [target])
    
    X, y = df[features], df[target]

    model = PyTorchMLPWrapper(
        hidden_size=best_params.get('hidden_size', 256),
        num_layers=best_params.get('num_layers', 3),
        dropout_rate=best_params.get('dropout_rate', 0.1),
        learning_rate=best_params.get('learning_rate', 0.001),
        batch_size=best_params.get('batch_size', 512),
        epochs=300, 
        patience=25,
        verbose=True
    )
    pipeline = Pipeline(steps=[('preprocessor', create_preprocessor()), ('model', model)])

    run_kfold_validation(pipeline, X, y, n_splits=5, output_dir=RESULTS_DIR)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    target_label = f"{target}_MLP"
    plot_feature_importance(pipeline, X_test, y_test, target_label, output_dir=RESULTS_DIR)
    plot_actual_vs_predicted(y_test, y_pred, target_label, output_dir=RESULTS_DIR)
    plot_error_distribution(y_test, y_pred, target_label, output_dir=RESULTS_DIR)

    saved_model_path = os.path.join(MODEL_DIR, "best_pipeline_mlp.joblib")
    joblib.dump(pipeline, saved_model_path)

if __name__ == "__main__":
    main()