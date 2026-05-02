import numpy as np
import os
from sklearn.model_selection import KFold, cross_validate, GroupKFold


def run_kfold_validation(pipeline, X, y, model_name, n_splits=5, output_dir="./"):
    '''
    Executa a validação cruzada K-Fold em um pipeline de aprendizado de máquina.
    Avalia as métricas RMSE, MAE e R2, e salva os resultados em um arquivo de texto.
    '''
    # Inicia o objeto K-Fold com embaralhamento dos dados
    print(f"Iniciando validação cruzada com {n_splits} folds")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Executa a validação cruzada para as métricas escolhidas
    cv_results = cross_validate(
        pipeline, X, y, cv=kf, 
        scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        return_train_score=False, n_jobs=1 
    )
    
    # Converte as métricas de erro para valores positivos
    rmse_scores = -cv_results['test_neg_root_mean_squared_error']
    mae_scores = -cv_results['test_neg_mean_absolute_error']
    r2_scores = cv_results['test_r2']
    
    # Calcula a média e o desvio padrão de cada métrica
    rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
    mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
    
    # Exibe os resultados no console
    print(f"RMSE: {rmse_mean:.2f} dB ± {rmse_std:.4f}")
    print(f"MAE:  {mae_mean:.2f} dB ± {mae_std:.4f}")
    print(f"R2:   {r2_mean:.4f} ± {r2_std:.4f}")
    
    # Salva as métricas calculadas em um arquivo txt
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.replace(' ', '')}_kfold_metrics.txt"
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(f"RMSE Médio: {rmse_mean:.2f} dB (Desvio padrão: {rmse_std:.4f})\n")
        f.write(f"MAE Médio:  {mae_mean:.2f} dB (Desvio padrão: {mae_std:.4f})\n")
        f.write(f"R2 Médio:   {r2_mean:.4f} (Desvio padrão: {r2_std:.4f})\n")

def run_gkfold_validation(pipeline, X, y, groups, model_name, n_splits=5, output_dir="./"):
    '''
    Executa a validação cruzada espacial Leave-Antenna-Out (GroupKFold).
    Avalia as métricas de generalização RMSE, MAE e R2 para áreas (antenas) não vistas.
    '''
    print(f"Iniciando validação cruzada espacial (GroupKFold) com {n_splits} folds")
    gkf = GroupKFold(n_splits=n_splits)
    
    cv_results = cross_validate(
        pipeline, X, y, groups=groups, cv=gkf, 
        scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        return_train_score=False, n_jobs=1 
    )
    
    rmse_scores = -cv_results['test_neg_root_mean_squared_error']
    mae_scores = -cv_results['test_neg_mean_absolute_error']
    r2_scores = cv_results['test_r2']
    
    rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
    mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
    
    print(f"GroupKFold RMSE: {rmse_mean:.2f} dB ± {rmse_std:.4f}")
    print(f"GroupKFold MAE:  {mae_mean:.2f} dB ± {mae_std:.4f}")
    print(f"GroupKFold R2:   {r2_mean:.4f} ± {r2_std:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.replace(' ', '')}_gkfold_metrics.txt"
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("Relatório de Generalização Espacial (GroupKFold por Antena)\n")
        f.write("="*60 + "\n")
        f.write(f"RMSE Médio: {rmse_mean:.2f} dB (Desvio padrão: {rmse_std:.4f})\n")
        f.write(f"MAE Médio:  {mae_mean:.2f} dB (Desvio padrão: {mae_std:.4f})\n")
        f.write(f"R2 Médio:   {r2_mean:.4f} (Desvio padrão: {r2_std:.4f})\n")