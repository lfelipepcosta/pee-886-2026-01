import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.inspection import permutation_importance

def plot_feature_importance(pipeline, X_val, y_val, target_name, output_dir):
    print("Gerando gráfico de Importância (Permutation Importance)...")
    result = permutation_importance(pipeline, X_val, y_val, n_repeats=3, random_state=42, n_jobs=1, scoring='neg_root_mean_squared_error')
    
    plt.figure(figsize=(12, 8))
    importances = pd.Series(result.importances_mean, index=X_val.columns)
    importances.nlargest(15).sort_values().plot(kind='barh', color='navy')
    plt.title(f'Importância Relativa por Reembaralhamento - {target_name}')
    plt.xlabel('Custo do Erro se a Feature for Falsificada (RMSE)')
    plt.ylabel('Features')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"feature_importance_{target_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_actual_vs_predicted(y_test, y_pred, target_name, output_dir):
    print("Gerando gráfico de Previsão vs Valores Reais...")
    plt.figure(figsize=(10, 8))
    
    sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
    y_test_vals = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
    
    plt.scatter(y_test_vals, y_pred[sample_idx], alpha=0.3, color='royalblue', s=12)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(f'Propagação Verdadeira x Sinal Previsto - {target_name}')
    plt.xlabel('Verdadeiro Sinal Medido Efetivo')
    plt.ylabel('Sinal Previsto')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"actual_vs_predicted_{target_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_error_distribution(y_test, y_pred, target_name, output_dir):
    print("Gerando gráfico de distribuição de erros (Resíduos)...")
    errors = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True, color='purple')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
    plt.title(f'Distribuição de Erro (Real vs Previsto) - {target_name}')
    plt.xlabel('Erro Residual (dB)')
    plt.ylabel('Frequência')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"error_distribution_{target_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_coverage_map(df_coverage, df_route, df_antennas, model_name, target_name, output_dir):
    print(f"Renderizando mapa de cobertura geográfico (PDF) para o modelo {model_name}...")
    
    coverage_gdf = gpd.GeoDataFrame(
        df_coverage, 
        geometry=[Point(xy) for xy in zip(df_coverage['Longitude'], df_coverage['Latitude'])], 
        crs="EPSG:4674"
    )

    fig, ax = plt.subplots(figsize=(16, 12))
    vmin_data = coverage_gdf['RSRP_dBm'].quantile(0.05)
    vmax_data = coverage_gdf['RSRP_dBm'].quantile(0.95)

    coverage_gdf.plot(
        column='RSRP_dBm', ax=ax, cmap='RdYlGn', vmin=vmin_data, vmax=vmax_data, 
        legend=True, markersize=2, alpha=0.8, legend_kwds={'label': 'RSRP_dBm Predito'}
    )
    
    if not df_route.empty:
        route_gdf = gpd.GeoDataFrame(df_route, geometry=[Point(xy) for xy in zip(df_route['Longitude'], df_route['Latitude'])], crs="EPSG:4674")
        route_gdf.plot(ax=ax, color='blue', markersize=0.5, alpha=0.4, label="Rota Drive Test Real")

    if not df_antennas.empty:
        antennas_gdf = gpd.GeoDataFrame(df_antennas, geometry=[Point(xy) for xy in zip(df_antennas['Antena_Lon'], df_antennas['Antena_Lat'])], crs="EPSG:4674")
        antennas_gdf.plot(ax=ax, color='black', marker='x', markersize=80, linewidths=1.5, label="Sites (Antenas)", zorder=10)

    plt.title(f"Mapa de Cobertura Espacial ({model_name}) - {target_name}", fontsize=16)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_path = os.path.join(output_dir, f"coverage_map_{target_name}_{model_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)