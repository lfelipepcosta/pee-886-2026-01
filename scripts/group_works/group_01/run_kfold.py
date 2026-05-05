import torch
import os
import json
import numpy as np
import sys

# Adiciona o diretório raiz ao path para permitir a importação do pacote qml
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(repo_root)

from qml.group_works.group_01.loaders.mri_loader import get_kfold_dataloaders
from qml.group_works.group_01.models.hybrid_resnet import HybridResNet18
from qml.group_works.group_01.models.classical_resnet import ClassicalResNet18
from qml.group_works.group_01.trainer.training_loop import train_model

def load_params(filename):
    path = os.path.join(repo_root, "data/group_works/group_01", filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def run_kfold_experiment(model_type, params, folds, device, output_dir):
    """
    Executa a validação cruzada K-Fold para o modelo selecionado.
    O K-Fold treina e avalia o modelo 'K' vezes, garantindo que o modelo 
    não decorou um conjunto específico de dados e tem uma performance estável.
    """
    import time
    print(f"\nIniciando K-Fold para: {model_type}")
    
    # Armazena as acurácias e tempos de cada uma das 5 rodadas (folds)
    accuracies = []
    times = []
    
    for i, (train_loader, val_loader) in enumerate(folds):
        print(f"--- Fold {i+1}/{len(folds)} ---")
        start_fold = time.time()
        
        # Instancia um modelo ZERADO a cada fold para não vazar conhecimento
        if model_type == "classical":
            model = ClassicalResNet18(num_classes=2)
        else:
            q_depth = params.get('q_depth', 1)
            model = HybridResNet18(num_classes=2, n_qubits=4, q_depth=q_depth)
            
        lr = params.get('lr', 0.0004)
        wd = params.get('weight_decay', 1e-4)
        
        # Treina o modelo nesta divisão específica de dados
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=f"KFold_{model_type}_Fold{i+1}",
            epochs=5, # Usamos 5 épocas no k-fold apenas para estimar a estabilidade rapidamente
            lr=lr,
            weight_decay=wd,
            device=device,
            output_dir=output_dir,
            verbose=False # Mantém o terminal limpo
        )
        
        fold_time = time.time() - start_fold
        times.append(fold_time)
        best_acc = max(history['val_acc'])
        accuracies.append(best_acc)
        print(f"Fold {i+1} Acc: {best_acc:.4f} | Tempo: {fold_time:.2f}s")
        
    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    avg_time = np.mean(times)
    
    print(f"\nResultado Final {model_type}: {avg_acc:.4f} +- {std_acc:.4f}")
    return {
        'model': model_type,
        'accuracies': accuracies,
        'times': times,
        'avg_acc': avg_acc,
        'std_acc': std_acc,
        'avg_time': avg_time
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Carrega parâmetros
    params_h = load_params('best_params_hybrid.json')
    params_c = load_params('best_params_classical.json')
    
    if not params_h or not params_c:
        print("Aviso: Hiperparametros otimizados nao encontrados. Usando padroes.")
        params_h = {'lr': 0.0004, 'weight_decay': 1e-4, 'q_depth': 1}
        params_c = {'lr': 0.0004, 'weight_decay': 1e-4}
        
    # Obtém folds
    folds = get_kfold_dataloaders(batch_size=16, n_splits=5)
    
    if not folds:
        print("Erro ao carregar dados.")
        return

    output_dir = os.path.join(repo_root, "data/group_works/group_01/kfold")
    os.makedirs(output_dir, exist_ok=True)

    # Roda experimentos e coleta resultados
    results = []
    results.append(run_kfold_experiment("classical", params_c, folds, device, output_dir))
    results.append(run_kfold_experiment("hybrid", params_h, folds, device, output_dir))
    
    # Salva relatório único consolidado
    report_path = os.path.join(output_dir, "kfold_report.txt")
    with open(report_path, "w") as f:
        f.write(f"RELATORIO CONSOLIDADO K-FOLD (5 Folds)\n")
        f.write("="*40 + "\n")
        for res in results:
            f.write(f"\nMODELO: {res['model'].upper()}\n")
            f.write(f"Acuracias: {res['accuracies']}\n")
            f.write(f"Tempos por Fold: {[f'{t:.2f}s' for t in res['times']]}\n")
            f.write(f"ACURACIA MEDIA: {res['avg_acc']:.4f} +- {res['std_acc']:.4f}\n")
            f.write(f"TEMPO MEDIO: {res['avg_time']:.2f}s\n")
            f.write("-" * 20 + "\n")
            
    print(f"\nRelatorio consolidado salvo em: {report_path}")

if __name__ == "__main__":
    main()
