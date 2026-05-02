import subprocess
import sys
import time
import os

# Define a raiz do repositório para centralizar os logs na pasta de dados
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
log_dir = os.path.join(repo_root, "data", "luiz_costa", "execution_reports")
os.makedirs(log_dir, exist_ok=True)

# Lista de scripts para execução do Pipeline ABSOLUTAMENTE COMPLETO (Baselines + Híbrido)
scripts = [
    # "run_optimization_dt.py",     # 1. Otimização de hiperparâmetros Decision Tree
    "run_main_dt.py",             # 2. Treinamento final e métricas Decision Tree
    "run_inference_dt.py",        # 3. Geração do mapa de cobertura Decision Tree
    
    # "run_optimization_xgb.py",    # 4. Otimização de hiperparâmetros XGBoost
    "run_main_xgb.py",            # 5. Treinamento final e métricas XGBoost
    "run_inference_xgb.py",       # 6. Geração do mapa de cobertura XGBoost
    
    # "run_optimization_mlp.py",    # 7. Otimização de hiperparâmetros MLP
    "run_main_mlp.py",            # 8. Treinamento final e métricas MLP
    "run_inference_mlp.py",       # 9. Geração do mapa de cobertura MLP
    
    # "run_optimization.py",        # 10. Otimização de hiperparâmetros Híbrido Quântico
    "run_main.py",                # 11. Treinamento final e métricas Híbrido
    "run_inference_hybrid.py",    # 12. Geração do mapa de cobertura Híbrido
    
    "run_spatial_validation.py"   # 13. Validação espacial (Comparação de todos os modelos)
]

# Configura o caminho completo do arquivo de log no diretório data
timestamp = time.strftime("%Y%m%d_%H%M")
log_file = os.path.join(log_dir, f"execution_report_hybrid_{timestamp}.txt")

print(f"Iniciando Orquestrador de Execução (Pipeline Híbrido Quântico)")
print(f"O relatório detalhado será salvo em: {log_file}\n")

with open(log_file, "w", encoding="utf-8") as f_log:
    total_start = time.time()
    f_log.write(f"Relatório de Execução - Pipeline Híbrido Quântico\n")
    f_log.write(f"Início: {time.ctime(total_start)}\n")
    f_log.write(f"{'='*60}\n")
    
    for script in scripts:
        print(f"{'='*60}")
        print(f"EXECUTANDO: {script}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            # Garante que as saídas do Python sejam exibidas em tempo real
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            # Executa o script e captura a saída
            process = subprocess.Popen(
                ["python3", "-u", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=os.path.dirname(__file__)
            )
            
            # Pipe da saída do subprocesso para o terminal principal
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
                sys.stdout.flush()
            
            process.stdout.close()
            return_code = process.wait()
            
            end_time = time.time()
            elapsed_seconds = end_time - start_time
            
            # Registra o status e o tempo de execução
            if return_code == 0:
                status = "SUCESSO"
                print(f"\n[OK] {script} concluído em {elapsed_seconds/60:.2f} minutos.")
            else:
                status = f"ERRO (Código {return_code})"
                print(f"\n[ERRO] {script} falhou após {elapsed_seconds/60:.2f} minutos.")
            
            f_log.write(f"{script}: {status} ({elapsed_seconds:.2f}s)\n")
            f_log.flush() 
            
        except KeyboardInterrupt:
            print("\nExecução interrompida pelo usuário.")
            f_log.write(f"{script}: INTERROMPIDO\n")
            sys.exit(1)
        except Exception as e:
            print(f"\nFalha crítica ao tentar rodar {script}: {e}")
            f_log.write(f"{script}: FALHA CRÍTICA ({e})\n")
            f_log.flush()

    total_end = time.time()
    total_time_min = (total_end - total_start) / 60
    
    completion_msg = (
        f"\n{'='*60}\n"
        f"PIPELINE DE BASELINES CONCLUÍDO!\n"
        f"Tempo total de execução: {total_time_min:.2f} minutos.\n"
        f"Relatório salvo em: {log_file}\n"
        f"{'='*60}\n"
    )
    print(completion_msg)
    f_log.write(f"{'='*60}\n")
    f_log.write(f"Fim: {time.ctime(total_end)}\n")
    f_log.write(f"Tempo Total: {total_time_min:.2f} min\n")