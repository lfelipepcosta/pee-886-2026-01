import subprocess
import sys
import time
import os
import statistics

# Define a raiz do repositório para centralizar os logs na pasta de dados
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
log_dir = os.path.join(repo_root, "data", "luiz_costa", "execution_reports")
os.makedirs(log_dir, exist_ok=True)

# Pipelines definidos por seus scripts
pipelines = {
    "DecisionTree": [
        "run_optimization_dt.py", 
        "run_main_dt.py", 
        "run_inference_dt.py"
    ],
    "XGBoost": [
        "run_optimization_xgb.py", 
        "run_main_xgb.py", 
        "run_inference_xgb.py"
    ],
    "MLP": [
        "run_optimization_mlp.py", 
        "run_main_mlp.py", 
        "run_inference_mlp.py"
    ],
    "Hibrido": [
        "run_optimization.py", 
        "run_main.py", 
        "run_inference_hybrid.py"
    ]
}

# Configura o caminho completo do arquivo de log no diretório data
timestamp = time.strftime("%Y%m%d_%H%M")
log_file = os.path.join(log_dir, f"benchmark_report_{timestamp}.txt")

print(f"Iniciando Orquestrador de Benchmark (10 execuções por pipeline)")
print(f"O relatório detalhado será salvo em: {log_file}\n")

# Dicionário para salvar os tempos totais de cada execução
execution_times = {
    "DecisionTree": [],
    "XGBoost": [],
    "MLP": [],
    "Hibrido": []
}

with open(log_file, "w", encoding="utf-8") as f_log:
    total_start = time.time()
    f_log.write(f"Relatório de Benchmark de Tempo de Execução\n")
    f_log.write(f"Início: {time.ctime(total_start)}\n")
    f_log.write(f"{'='*60}\n\n")
    
    # Executa num_rounds iterações do Benchmark
    num_rounds = 5
    for run in range(1, num_rounds + 1):
        print(f"\n{'#'*60}")
        print(f"### INICIANDO RODADA {run}/{num_rounds} ###")
        print(f"{'#'*60}\n")
        f_log.write(f"--- RODADA {run}/{num_rounds} ---\n")
        
        for pipe_name, scripts in pipelines.items():
            print(f"\n[Pipeline {pipe_name}] - Rodada {run}")
            pipe_start_time = time.time()
            
            for script in scripts:
                print(f"{'='*40}")
                print(f"EXECUTANDO: {script}")
                print(f"{'='*40}\n")
                
                script_start_time = time.time()
                
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
                    
                    script_end_time = time.time()
                    elapsed_script = script_end_time - script_start_time
                    
                    if return_code == 0:
                        status = "SUCESSO"
                        print(f"\n[OK] {script} concluído em {elapsed_script/60:.2f} minutos.")
                    else:
                        status = f"ERRO (Código {return_code})"
                        print(f"\n[ERRO] {script} falhou após {elapsed_script/60:.2f} minutos.")
                    
                    f_log.write(f"R{run} | {script}: {status} ({elapsed_script:.2f}s)\n")
                    f_log.flush() 
                    
                except KeyboardInterrupt:
                    print("\nExecução interrompida pelo usuário.")
                    f_log.write(f"R{run} | {script}: INTERROMPIDO\n")
                    sys.exit(1)
                except Exception as e:
                    print(f"\nFalha crítica ao tentar rodar {script}: {e}")
                    f_log.write(f"R{run} | {script}: FALHA CRÍTICA ({e})\n")
                    f_log.flush()
            
            pipe_end_time = time.time()
            elapsed_pipe = pipe_end_time - pipe_start_time
            execution_times[pipe_name].append(elapsed_pipe)
            f_log.write(f"--> Tempo Total Pipeline {pipe_name} (Rodada {run}): {elapsed_pipe:.2f}s ({elapsed_pipe/60:.2f} min)\n\n")
            f_log.flush()

        # Executa a validação espacial ao final das execuções da rodada
        print(f"{'='*40}")
        print(f"EXECUTANDO VALIDAÇÃO ESPACIAL: run_spatial_validation_benchmark.py")
        print(f"{'='*40}\n")
        
        val_start = time.time()
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["BENCHMARK_ROUND"] = str(run)
            process = subprocess.Popen(
                ["python3", "-u", "run_spatial_validation_benchmark.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=os.path.dirname(__file__)
            )
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
                sys.stdout.flush()
            process.stdout.close()
            return_code = process.wait()
            val_end = time.time()
            elapsed_val = val_end - val_start
            
            status = "SUCESSO" if return_code == 0 else f"ERRO (Código {return_code})"
            f_log.write(f"R{run} | run_spatial_validation.py: {status} ({elapsed_val:.2f}s)\n")
            
            # Lê os tempos individuais do arquivo temporário
            temp_file = os.path.join(repo_root, "data", "luiz_costa", "validation_time_temp.txt")
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as f_temp:
                    for line in f_temp:
                        if ":" in line:
                            model_n, t_val = line.strip().split(":")
                            f_log.write(f"R{run} | Tempo Validação {model_n.upper()}: {float(t_val):.2f}s\n")
            
            f_log.write("\n")
            f_log.flush()
        except Exception as e:
            f_log.write(f"R{run} | run_spatial_validation.py: FALHA CRÍTICA ({e})\n\n")
            f_log.flush()

    # Seção final de estatísticas
    total_end = time.time()
    total_time_min = (total_end - total_start) / 60
    
    f_log.write(f"{'='*60}\n")
    f_log.write(f"RESULTADOS FINAIS DO BENCHMARK\n")
    f_log.write(f"{'='*60}\n\n")
    
    for pipe_name, times in execution_times.items():
        f_log.write(f"Pipeline: {pipe_name}\n")
        f_log.write(f"Tempos (em segundos):\n")
        for t in times:
            f_log.write(f"  - {t:.2f}s ({t/60:.2f} min)\n")
            
        mean_time = statistics.mean(times)
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
        
        f_log.write(f"Média: {mean_time:.2f}s ({mean_time/60:.2f} min)\n")
        f_log.write(f"Desvio Padrão: {stdev_time:.2f}s ({stdev_time/60:.2f} min)\n")
        f_log.write(f"{'-'*40}\n\n")
        
    f_log.write(f"{'='*60}\n")
    f_log.write(f"Fim do Benchmark: {time.ctime(total_end)}\n")
    f_log.write(f"Tempo Total Geral: {total_time_min:.2f} min\n")
    
    completion_msg = (
        f"\n{'='*60}\n"
        f"BENCHMARK CONCLUÍDO COM SUCESSO!\n"
        f"Tempo total de execução geral: {total_time_min:.2f} minutos.\n"
        f"Relatório final consolidado salvo em: {log_file}\n"
        f"{'='*60}\n"
    )
    print(completion_msg)
