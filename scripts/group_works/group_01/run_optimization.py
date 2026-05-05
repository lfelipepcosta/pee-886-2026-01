import optuna
import torch
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# Adiciona a raiz do repositório ao path para permitir imports dos módulos QML
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(repo_root)

from qml.group_works.group_01.loaders.mri_loader import get_dataloaders
from qml.group_works.group_01.models.hybrid_resnet import HybridResNet18
from qml.group_works.group_01.models.classical_resnet import ClassicalResNet18
from qml.group_works.group_01.trainer.training_loop import train_model

def optuna_objective(trial, model_type, train_loader, val_loader, device):
    """
    Função de custo para maximizar a acurácia no Optuna.
    """
    # Hiperparâmetros a serem otimizados
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    if model_type == "classical":
        model = ClassicalResNet18(num_classes=2)
        model_name = "Optuna_Classical"
    else:
        # No modelo híbrido, também otimizamos a profundidade (camadas) do circuito quântico!
        q_depth = trial.suggest_int('q_depth', 1, 3)
        model = HybridResNet18(num_classes=2, n_qubits=4, q_depth=q_depth)
        model_name = "Optuna_Hybrid"
        
    # Define diretório para o melhor modelo temporário da otimização
    optuna_dir = os.path.join(repo_root, "data/group_works/group_01/optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    
    # Treino com verbose=False para não poluir o terminal
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=f"Optuna_{model_type}_Best",
        epochs=5,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        output_dir=optuna_dir,
        verbose=False
    )
    
    # Optuna maximiza a melhor acurácia de validação atingida durante as épocas
    best_acc = max(history['val_acc'])
    
    # Se este trial for o melhor até agora, o checkpoint já foi salvo pelo train_model 
    # No entanto, o train_model salva o melhor DAQUELA RODADA. 
    # Para garantir que temos o melhor global do Optuna, vamos gerenciar aqui
    
    study = trial.study
    is_best = False
    try:
        if best_acc > study.best_value:
            is_best = True
    except ValueError:
        is_best = True # Primeiro trial
    
    if is_best:
        # Renomeia para um arquivo definitivo do melhor trial
        final_best_path = os.path.join(optuna_dir, f"best_{model_type}_checkpoint.pth")
        current_pth = os.path.join(optuna_dir, f"Optuna_{model_type}_Best_best.pth")
        if os.path.exists(current_pth):
            import shutil
            shutil.copy(current_pth, final_best_path)
    
    # Print customizado para o usuário acompanhar o progresso sem spam
    study = trial.study
    try:
        current_best = study.best_value
        current_best = max(current_best, best_acc)
    except ValueError:
        current_best = best_acc
        
    print(f"Trial {trial.number} finalizado | Acc: {best_acc:.4f} | Melhor ate agora: {current_best:.4f}")
    
    return best_acc

def main():
    import time
    start_total = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando Otimizacao Optuna usando: {device}")
    
    # Agora recebemos 3 loaders: Treino, Validação e Teste (Cego)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)
    
    # 1. Otimização do Modelo Clássico
    print("\n--- Otimizando ClassicalResNet18 ---")
    study_classical = optuna.create_study(direction="maximize", study_name="MRI_Classical")
    study_classical.optimize(lambda trial: optuna_objective(trial, "classical", train_loader, val_loader, device), n_trials=10)
    
    # 2. Otimização do Modelo Híbrido
    print("\n--- Otimizando HybridResNet18 ---")
    study_hybrid = optuna.create_study(direction="maximize", study_name="MRI_Hybrid")
    study_hybrid.optimize(lambda trial: optuna_objective(trial, "hybrid", train_loader, val_loader, device), n_trials=10)
    
    # 3. Salvar Melhores Parâmetros e Tempos
    output_dir = os.path.join(repo_root, "data", "group_works", "group_01")
    os.makedirs(output_dir, exist_ok=True)
    
    total_time = time.time() - start_total
    
    with open(os.path.join(output_dir, "best_params_classical.json"), "w") as f:
        json.dump(study_classical.best_params, f, indent=4)
        
    with open(os.path.join(output_dir, "best_params_hybrid.json"), "w") as f:
        json.dump(study_hybrid.best_params, f, indent=4)
        
    with open(os.path.join(output_dir, "optimization_time.txt"), "w") as f:
        f.write(f"Tempo total de otimizacao: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")

    print(f"\nOtimizacao Concluida!")
    print(f"Melhor Classico: {study_classical.best_value:.4f}")
    print(f"Melhor Hibrido: {study_hybrid.best_value:.4f}")

if __name__ == "__main__":
    main()
