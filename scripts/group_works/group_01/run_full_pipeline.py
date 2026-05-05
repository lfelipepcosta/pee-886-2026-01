import os
import sys
import json
import joblib
import torch
import numpy as np
from datetime import datetime

# Adiciona a raiz do repositório ao path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(repo_root)

from qml.group_works.group_01.loaders.mri_loader import get_dataloaders, get_kfold_dataloaders
from qml.group_works.group_01.models.hybrid_resnet import HybridResNet18
from qml.group_works.group_01.models.classical_resnet import ClassicalResNet18
from qml.group_works.group_01.trainer.training_loop import train_model, test_model
from qml.group_works.group_01.evaluation.metrics import plot_comparison

from run_optimization import main as run_opt
from run_kfold import run_kfold_experiment

def save_model_joblib(model, config, params, path):
    """
    Salva o modelo e seus metadados usando joblib
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'hyperparams': params,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(save_dict, path)
    print(f"Modelo salvo em: {path}")

def main():
    import time
    start_total = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INICIANDO PIPELINE COMPLETO - {datetime.now()}")
    print(f"Dispositivo: {device}")

    base_dir = os.path.join(repo_root, "data/group_works/group_01")
    results_dir = os.path.join(base_dir, "pipeline_results")
    model_dir = os.path.join(base_dir, "trained_models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Otimização com Optuna
    print("\nOtimizacao de hiperparametros")
    run_opt() 

    # Carrega os melhores parâmetros gerados
    def load_json(name):
        path = os.path.join(base_dir, name)
        with open(path, 'r') as f: return json.load(f)
    
    best_h = load_json('best_params_hybrid.json')
    best_c = load_json('best_params_classical.json')

    # Validação cruzada com K-Fold
    print("\nValidação cruzada com K-Fold")
    kfold_dir = os.path.join(base_dir, "kfold")
    os.makedirs(kfold_dir, exist_ok=True)
    folds = get_kfold_dataloaders(batch_size=16, n_splits=5)
    run_kfold_experiment("classical", best_c, folds, device, kfold_dir)
    run_kfold_experiment("hybrid", best_h, folds, device, kfold_dir)

    # Treinamento final e teste cego
    print("\nTreinamento final e teste cego")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

    # Treino híbrido final
    h_model = HybridResNet18(num_classes=2, n_qubits=4, q_depth=best_h['q_depth'])
    h_history = train_model(
        h_model, train_loader, val_loader, 
        model_name="Hybrid_Final", epochs=20, 
        lr=best_h['lr'], weight_decay=best_h['weight_decay'], 
        device=device, output_dir=results_dir, verbose=False
    )
    h_acc, h_preds, h_true = test_model(h_model, test_loader, device=device)

    # Treino clássico final
    c_model = ClassicalResNet18(num_classes=2)
    c_history = train_model(
        c_model, train_loader, val_loader, 
        model_name="Classical_Final", epochs=20, 
        lr=best_c['lr'], weight_decay=best_c['weight_decay'], 
        device=device, output_dir=results_dir, verbose=False
    )
    c_acc, c_preds, c_true = test_model(c_model, test_loader, device=device)

    # Salva modelos e gráficos
    save_model_joblib(h_model, {'n_qubits':4, 'q_depth':best_h['q_depth']}, best_h, os.path.join(model_dir, "hybrid_model.joblib"))
    save_model_joblib(c_model, {}, best_c, os.path.join(model_dir, "classical_model.joblib"))
    
    from qml.group_works.group_01.evaluation.metrics import plot_confusion_matrix
    plot_comparison(h_history, c_history, output_dir=results_dir)
    plot_confusion_matrix(h_true, h_preds, "Hybrid", results_dir)
    plot_confusion_matrix(c_true, c_preds, "Classical", results_dir)

    # Relatório Final
    total_time = time.time() - start_total
    report_path = os.path.join(results_dir, "pipeline_summary.txt")
    with open(report_path, "w") as f:
        f.write(f"Sumario do Pipeline - {datetime.now()}\n")
        f.write(f"Tempo Total: {total_time/60:.2f} minutos\n")
        f.write(f"Acuracia Teste Hibrido: {h_acc:.4f}\n")
        f.write(f"Acuracia Teste Classico: {c_acc:.4f}\n")
        f.write(f"Hiperparametros H: {best_h}\n")
        f.write(f"Hiperparametros C: {best_c}\n")

    print(f"\nPipeline concluído! Resultados em: {results_dir}")

if __name__ == "__main__":
    main()
