import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error
import time
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(repo_root)

BASE_PATH = "/home/luiz.costa/anatel_data/_mapas"
DT_PATH = os.path.join(BASE_PATH, "Dados_DT", "DT_Ntero_5G.csv")
PATH_CLUTTER = os.path.join(BASE_PATH, "RJ", "RJ", "RJ_Clutter_v5.tif")
PATH_ELEVATION = os.path.join(BASE_PATH, "RJ", "RJ", "RJ_Heights.tif")
MAX_DISTANCE_M = 40.0

def validate_ai_coverage(parquet_path, dt_path, output_dir, model_name):
    """
    Realiza a validação espacial cruzando os resultados de inferência da IA (Parquet)
    com os dados reais do Drive Test, aplicando limites de distância máxima.
    Gera as métricas estatísticas e os correspondentes gráficos de erro em PDF vetorizado.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Iniciando Validação Espacial para: {model_name} ---")
    print(f"Carregando Drive Test: {dt_path}")
    dt_df = pd.read_csv(dt_path, sep=None, engine='python')
    dt_df = dt_df.dropna(subset=['Latitude', 'Longitude'])
    dt_df = dt_df.rename(columns={'SS-RSRP': 'DT_RSRP'})
    dt_df = dt_df.dropna(subset=['DT_RSRP'])
    
    dt_geometry = [Point(xy) for xy in zip(dt_df['Longitude'], dt_df['Latitude'])]
    dt_gdf = gpd.GeoDataFrame(dt_df, geometry=dt_geometry, crs="EPSG:4674").to_crs(epsg=31983)

    print(f"Carregando dados simulados da IA: {parquet_path}")
    if not os.path.exists(parquet_path):
        print(f"Erro: arquivo Parquet não encontrado ({parquet_path}).")
        return
        
    sim_df = pd.read_parquet(parquet_path)
    sim_geometry = [Point(xy) for xy in zip(sim_df['Longitude'], sim_df['Latitude'])]
    sim_gdf = gpd.GeoDataFrame(sim_df, geometry=sim_geometry, crs="EPSG:4674").to_crs(epsg=31983)

    print("Executando junção espacial (Simulado vs Real)")
    validation_gdf = gpd.sjoin_nearest(dt_gdf, sim_gdf, how='inner', distance_col='match_distance_m', max_distance=MAX_DISTANCE_M)
    validation_gdf = validation_gdf.drop_duplicates(subset=['geometry'])

    total_matches = len(validation_gdf)
    print(f"Pontos alinhados geograficamente: {total_matches}")
    if total_matches == 0: return
    
    strict_gdf = validation_gdf[validation_gdf['match_distance_m'] <= MAX_DISTANCE_M].copy()

    print("Extraindo elevação e classes de terreno para análise espacial")
    with rasterio.open(PATH_ELEVATION) as src_elev:
        coords_elev = [(pt.x, pt.y) for pt in strict_gdf.to_crs(src_elev.crs).geometry]
        strict_gdf['Elevation'] = [val[0] for val in src_elev.sample(coords_elev)]

    with rasterio.open(PATH_CLUTTER) as src_clut:
        coords_clut = [(pt.x, pt.y) for pt in strict_gdf.to_crs(src_clut.crs).geometry]
        strict_gdf['Clutter'] = [val[0] for val in src_clut.sample(coords_clut)]

    actual_col = "DT_RSRP"
    sim_col = "RSRP_dBm"

    raw_actual, raw_predicted = validation_gdf[actual_col].values, validation_gdf[sim_col].values
    raw_errors = raw_predicted - raw_actual
    raw_rmse, raw_mse = np.sqrt(mean_squared_error(raw_actual, raw_predicted)), mean_squared_error(raw_actual, raw_predicted)
    raw_bias = np.mean(raw_errors)
    
    strict_actual, strict_predicted = strict_gdf[actual_col].values, strict_gdf[sim_col].values
    strict_errors = strict_predicted - strict_actual
    strict_rmse, strict_mse = np.sqrt(mean_squared_error(strict_actual, strict_predicted)), mean_squared_error(strict_actual, strict_predicted)
    strict_bias = np.mean(strict_errors)
    
    calib_predicted = strict_predicted - strict_bias
    calib_errors = calib_predicted - strict_actual
    calib_rmse, calib_mse = np.sqrt(mean_squared_error(strict_actual, calib_predicted)), mean_squared_error(strict_actual, calib_predicted)

    report = (
        f"Relatório de Validação da IA ({model_name}) (Malha 30m)\n"
        f"Dados Brutos ({len(validation_gdf)} pontos alinhados)\n"
        f"RMSE : {raw_rmse:.2f} dB\n"
        f"MSE  : {raw_mse:.2f} dB\n"
        f"Viés : {raw_bias:.2f} dB\n\n"
        f"Dados Restritos (Distância <= {MAX_DISTANCE_M}m, {len(strict_gdf)} pontos)\n"
        f"RMSE : {strict_rmse:.2f} dB\n"
        f"MSE  : {strict_mse:.2f} dB\n"
        f"Viés : {strict_bias:.2f} dB\n\n"
        f"Dados Calibrados (Ajustado pelo viés de {strict_bias:.2f} dB)\n"
        f"RMSE Calibrado : {calib_rmse:.2f} dB\n"
        f"MSE Calibrado  : {calib_mse:.2f} dB\n"
    )

    with open(f"{output_dir}/ai_validation_metrics_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 1. Gráficos Analíticos
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    plt.subplots_adjust(hspace=0.3)
    plot_configs = [
        (raw_actual, raw_predicted, raw_errors, raw_rmse, "Dados Brutos"),
        (strict_actual, calib_predicted, calib_errors, calib_rmse, "Dados Calibrados")
    ]
    for i, (act, pred, err, rmse, title) in enumerate(plot_configs):
        ax_scat, ax_hist = axes[0, i], axes[1, i]
        ax_scat.scatter(act, pred, alpha=0.3, color='blue', s=10)
        min_v, max_v = min(act.min(), pred.min()), max(act.max(), pred.max())
        ax_scat.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
        ax_scat.set_title(f"{title}\nRMSE: {rmse:.2f} dB", fontsize=16, fontweight='bold')
        ax_scat.set_xlabel("Medição DT RSRP [dBm]", fontsize=20)
        ax_scat.set_ylabel("Predito RSRP [dBm]", fontsize=20)
        ax_scat.grid(True, linestyle='--', alpha=0.5)

        ax_hist.hist(err, bins=50, color='orange', edgecolor='black', alpha=0.7)
        ax_hist.axvline(x=0, color='red', linestyle='dashed', linewidth=2)
        ax_hist.set_title("Distribuição do Erro", fontsize=16)
        ax_hist.set_xlabel("Erro (Predito - Real) [dB]", fontsize=20)
        ax_hist.set_ylabel("Frequência", fontsize=20)
        ax_hist.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{output_dir}/ai_validation_plots_{model_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. CDF do Erro Absoluto
    plt.figure(figsize=(10, 7))
    for err, label, color in zip([raw_errors, calib_errors], ['Bruto', 'Calibrado'], ['gray', 'green']):
        abs_err = np.abs(err)
        sorted_err = np.sort(abs_err)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        plt.plot(sorted_err, cdf, label=label, color=color, linewidth=2)
    plt.axhline(y=0.90, color='red', linestyle='dotted', alpha=0.7, label='Percentil 90')
    plt.title(f"CDF do Erro Absoluto - {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Erro Absoluto |dB|", fontsize=12)
    plt.ylabel("Probabilidade Cumulativa", fontsize=12)
    plt.xlim([0, 50])
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize='large')
    plt.savefig(f"{output_dir}/ai_validation_cdf_{model_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Espalhamento 3D e Análise Geográfica
    fig3d = plt.figure(figsize=(24, 8))
    ax1 = fig3d.add_subplot(131, projection='3d')
    sc3d = ax1.scatter(strict_gdf['Elevation'], strict_gdf['Clutter'], calib_errors, c=calib_errors, cmap='coolwarm', vmin=-30, vmax=30, alpha=0.6)
    ax1.set_xlabel('Elevação [m]'); ax1.set_ylabel('Classe Terreno'); ax1.set_zlabel('Erro Calibrado [dB]')
    ax1.set_title('Espalhamento 3D: Erro vs Ambiente', fontweight='bold')
    fig3d.colorbar(sc3d, ax=ax1, pad=0.1, fraction=0.02).set_label('Erro Ocorrido [dB]')

    ax2 = fig3d.add_subplot(132)
    ax2.scatter(strict_gdf['Elevation'], calib_errors, alpha=0.3, color='purple', s=10)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Elevação [m]'); ax2.set_ylabel('Erro Calibrado [dB]')
    ax2.set_title('Erro por Altitude', fontweight='bold'); ax2.grid(True, linestyle='--', alpha=0.5)

    ax3 = fig3d.add_subplot(133)
    ax3.scatter(strict_gdf['Clutter'], calib_errors, alpha=0.3, color='teal', s=10)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    
    dicionario_clutter = {0.0: "Sem dados", 1.0: "Urbana M. Alta", 2.0: "Urbana Alta", 3.0: "Urbana Media", 4.0: "Urbana baixa", 5.0: "Suburbana", 6.0: "Sub. Desordenada", 7.0: "Urbana Arborizada", 8.0: "Urbana Aberta", 11.0: "Aberta", 14.0: "Veg. Alta", 15.0: "Veg. Media", 16.0: "Veg. Baixa", 17.0: "Pastagem", 21.0: "Água"}
    unique_clutters = sorted(strict_gdf['Clutter'].dropna().unique())
    labels = [dicionario_clutter.get(val, str(val)) for val in unique_clutters]
    ax3.set_xticks(unique_clutters)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.set_xlabel('Classe de Clutter'); ax3.set_ylabel('Erro Calibrado [dB]')
    ax3.set_title('Erro por Classe de Terreno', fontweight='bold'); ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ai_validation_env_merit_{model_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig3d)
    
    print(f"Processo de validação concluído em {(time.time() - start_time) / 60.0:.2f} minutos.")

def main():
    models = ["xgb", "mlp", "hybrid"]
    for model in models:
        parquet_file = os.path.join(repo_root, "data", "luiz_costa", "inference_results", f"ai_coverage_30m_{model}.parquet")
        output_dir = os.path.join(repo_root, "data", "luiz_costa", f"spatial_validation_{model}")
        if os.path.exists(parquet_file):
            validate_ai_coverage(parquet_file, DT_PATH, output_dir, model)
        else:
            print(f"Parquet não encontrado para {model}, ignorando...")

if __name__ == "__main__":
    main()