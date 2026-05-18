import os
import sys
import time
import json
import tracemalloc
import torch

# Adiciona a raiz do repositório ao path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(repo_root)

from qml.group_works.group_01.loaders.mri_loader import get_dataloaders
from qml.group_works.group_01.models.hybrid_resnet_benchmark import HybridResNet18Benchmark
from qml.group_works.group_01.trainer.training_loop import train_model, test_model

def load_best_params(filename):
    path = os.path.join(repo_root, "data/group_works/group_01", filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {'lr': 0.0004, 'weight_decay': 1e-4, 'q_depth': 1}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Os métodos de diferenciação que o PennyLane suporta
    # Colocamos o parameter-shift primeiro como pedido
    diff_methods = ["parameter-shift", "adjoint", "backprop", "finite-diff"]
    
    # Busca dataloaders para treino, val e teste (sem K-Fold)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)
    params = load_best_params('best_params_hybrid.json')

    output_dir = os.path.join(repo_root, "data/group_works/group_01/diff_methods/benchmark_2nd_round")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "benchmark_report.txt")
    
    # Limpa/Cria o arquivo de relatório
    with open(report_path, "w") as f:
        f.write("RELATORIO DE BENCHMARK - METODOS DE DIFERENCIACAO (TESTE CEGO)\n")
        f.write("="*60 + "\n")

    for method in diff_methods:
        print(f"\n{'='*40}")
        print(f"[{method.upper()}] Inicializando treinamento...")
        print(f"{'='*40}")
        
        # Inicia a medição de memória
        tracemalloc.start()
        start_time = time.time()
        
        # Instancia o modelo com o método de diferenciação específico
        model = HybridResNet18Benchmark(
            num_classes=2, 
            n_qubits=4, 
            q_depth=params.get('q_depth', 1), 
            diff_method=method
        )
        
        # Treinamento de validação tradicional + teste cego (1 round)
        # Reduzimos as épocas para 5 só para o benchmark não levar dias, 
        # mas você pode alterar para as 20 do pipeline normal se preferir.
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=f"Hybrid_{method}",
            epochs=5,
            lr=params.get('lr', 0.0004),
            weight_decay=params.get('weight_decay', 1e-4),
            device=device,
            output_dir=output_dir,
            verbose=True
        )
        
        # Teste cego
        print(f"[{method.upper()}] Realizando Teste Cego...")
        test_acc, test_preds, test_true = test_model(model, test_loader, device=device)
        
        exec_time = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Conversão para MB
        peak_mem_mb = peak_mem / 1024 / 1024
        
        print(f"\n>>> [{method.upper()}] Concluido em {exec_time:.2f}s | Pico de Memória: {peak_mem_mb:.2f} MB | Acc Teste: {test_acc:.4f}\n")
        
        # Salva no relatório
        with open(report_path, "a") as f:
            f.write(f"\nMETODO: {method.upper()}\n")
            f.write(f"ACURACIA TESTE (BLIND): {test_acc:.4f}\n")
            f.write(f"TEMPO DE TREINO+TESTE (5 epocas): {exec_time:.2f}s\n")
            f.write(f"PICO DE MEMORIA RAM: {peak_mem_mb:.2f} MB\n")
            f.write("-" * 40 + "\n")

if __name__ == "__main__":
    main()
