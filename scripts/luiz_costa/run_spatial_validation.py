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

# Adiciona a raiz do repositório ao path para permitir imports dos módulos QML
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(repo_root)

# Caminhos base para arquivos geográficos, drive tests e rasters de altimetria
BASE_PATH = "/home/luiz.costa/anatel_data/_mapas"
DT_PATH = os.path.join("/home/luiz.costa/anatel_data/qml_data", "dt_5g_blind_test.csv")
PATH_CLUTTER = os.path.join(BASE_PATH, "RJ", "RJ", "RJ_Clutter_v5.tif")
PATH_ELEVATION = os.path.join(BASE_PATH, "RJ", "RJ", "RJ_Heights.tif")
MAX_DISTANCE_M = 40.0

def validate_ai_coverage(parquet_path, dt_path, output_dir, model_name):
    '''
    Avalia a precisão do mapeamento comparando predições (IA) com medições reais (Base Teórica).
    Cruza dados geoespaciais e gera métricas de erro estatístico e gráficos analíticos.
    '''
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Iniciando validação espacial para {model_name}")
    print(f"Carregando Drive Test real em {dt_path}")
    
    # Carrega dados do mundo real filtrando apenas coordenadas e sinal RSRP
    dt_df = pd.read_csv(dt_path, sep=None, engine='python')
    dt_df = dt_df.dropna(subset=['Latitude', 'Longitude'])
    dt_df = dt_df.rename(columns={'SS-RSRP': 'DT_RSRP'})
    dt_df = dt_df.dropna(subset=['DT_RSRP'])
    
    # Converte coordenadas geográficas para projetadas (UTM) para cálculo de distância em metros
    dt_geometry = [Point(xy) for xy in zip(dt_df['Longitude'], dt_df['Latitude'])]
    dt_gdf = gpd.GeoDataFrame(dt_df, geometry=dt_geometry, crs="EPSG:4674").to_crs(epsg=31983)

    print(f"Lendo resultados de inferência da IA em {parquet_path}")
    if not os.path.exists(parquet_path):
        print(f"Erro: arquivo Parquet não encontrado ({parquet_path})")
        return
        
    # Carrega as predições geradas pela rede neural
    sim_df = pd.read_parquet(parquet_path)
    sim_geometry = [Point(xy) for xy in zip(sim_df['Longitude'], sim_df['Latitude'])]
    sim_gdf = gpd.GeoDataFrame(sim_df, geometry=sim_geometry, crs="EPSG:4674").to_crs(epsg=31983)
    
    # Mantém apenas a predição e a geometria para evitar conflito de nomes de colunas no SJOIN
    sim_gdf = sim_gdf[['RSRP_dBm', 'geometry']]

    print("Realizando cruzamento espacial por proximidade (SJOIN)")
    # Une os pontos do Drive Test aos pontos da grade virtual mais próximos
    validation_gdf = gpd.sjoin_nearest(dt_gdf, sim_gdf, how='inner', distance_col='match_distance_m', max_distance=MAX_DISTANCE_M)
    
    # Agrupa por coordenadas e calcula a média do sinal real e predito para pontos repetidos
    print("Agrupando medições repetidas por coordenada (Média de sinal)")
    
    # Usamos as colunas originais do dt_gdf para o agrupamento
    group_cols = ['Latitude', 'Longitude']
    validation_gdf = validation_gdf.groupby(group_cols, as_index=False).agg({
        'DT_RSRP': 'mean',
        'RSRP_dBm': 'mean',
        'match_distance_m': 'mean'
    })
    
    # Reconverte para GeoDataFrame após o agrupamento para manter as propriedades espaciais
    val_geometry = [Point(xy) for xy in zip(validation_gdf['Longitude'], validation_gdf['Latitude'])]
    validation_gdf = gpd.GeoDataFrame(validation_gdf, geometry=val_geometry, crs="EPSG:4674").to_crs(epsg=31983)

    total_matches = len(validation_gdf)
    print(f"Pontos comparados com sucesso: {total_matches}")
    if total_matches == 0: return
    
    # Filtra pontos que ficaram dentro do raio aceitável para validação científica
    strict_gdf = validation_gdf[validation_gdf['match_distance_m'] <= MAX_DISTANCE_M].copy()

    print("Extraindo informações de elevação e clutter para os pontos cruzados")
    # Amostra dados físicos do terreno para correlacionar o erro com o ambiente
    with rasterio.open(PATH_ELEVATION) as src_elev:
        coords_elev = [(pt.x, pt.y) for pt in strict_gdf.to_crs(src_elev.crs).geometry]
        strict_gdf['Elevation'] = [val[0] for val in src_elev.sample(coords_elev)]

    with rasterio.open(PATH_CLUTTER) as src_clut:
        coords_clut = [(pt.x, pt.y) for pt in strict_gdf.to_crs(src_clut.crs).geometry]
        strict_gdf['Clutter'] = [val[0] for val in src_clut.sample(coords_clut)]

    actual_col = "DT_RSRP"
    sim_col = "RSRP_dBm"

    # Calcula métricas para os dados cruzados totais
    raw_actual, raw_predicted = validation_gdf[actual_col].values, validation_gdf[sim_col].values
    raw_errors = raw_predicted - raw_actual
    raw_rmse, raw_mse = np.sqrt(mean_squared_error(raw_actual, raw_predicted)), mean_squared_error(raw_actual, raw_predicted)
    raw_bias = np.mean(raw_errors)
    raw_std = np.std(raw_errors)
    
    # Calcula métricas estritas após filtragem de distância
    strict_actual, strict_predicted = strict_gdf[actual_col].values, strict_gdf[sim_col].values
    strict_errors = strict_predicted - strict_actual
    strict_rmse, strict_mse = np.sqrt(mean_squared_error(strict_actual, strict_predicted)), mean_squared_error(strict_actual, strict_predicted)
    strict_bias = np.mean(strict_errors)
    strict_std = np.std(strict_errors)
    
    # Calcula métricas calibradas compensando o viés (Bias) médio
    calib_predicted = strict_predicted - strict_bias
    calib_errors = calib_predicted - strict_actual
    calib_rmse, calib_mse = np.sqrt(mean_squared_error(strict_actual, calib_predicted)), mean_squared_error(strict_actual, calib_predicted)
    calib_std = np.std(calib_errors)

    # Monta o relatório de métricas detalhado
    num_blind_antennas = len(dt_df[['Antena_Lat', 'Antena_Lon']].drop_duplicates()) if 'Antena_Lat' in dt_df.columns else "Desconhecido"
    
    report = (
        f"Relatório de Validação de Campo IA ({model_name}) (Grade 30m)\n"
        f"--- INFORMAÇÕES DO BLIND TEST ---\n"
        f"Antenas Ocultas (Blind Test): {num_blind_antennas}\n"
        f"Pontos Totais de Blind Test: {len(dt_df)}\n"
        f"---------------------------------\n"
        f"Dados Totais ({len(validation_gdf)} pontos alinhados)\n"
        f"RMSE : {raw_rmse:.2f} dB\n"
        f"MSE  : {raw_mse:.2f} dB\n"
        f"Erro Médio (Viés ± Desvio Padrão): {raw_bias:.2f} ± {raw_std:.2f} dB\n\n"
        f"Dados Filtrados (Distância <= {MAX_DISTANCE_M}m, {len(strict_gdf)} pontos)\n"
        f"RMSE : {strict_rmse:.2f} dB\n"
        f"MSE  : {strict_mse:.2f} dB\n"
        f"Erro Médio (Viés ± Desvio Padrão): {strict_bias:.2f} ± {strict_std:.2f} dB\n\n"
        f"Dados Calibrados (Compensação de {strict_bias:.2f} dB)\n"
        f"RMSE Calibrado : {calib_rmse:.2f} dB\n"
        f"MSE Calibrado  : {calib_mse:.2f} dB\n"
        f"Erro Médio Calib. (Viés ± Desvio Padrão): 0.00 ± {calib_std:.2f} dB\n"
    )

    # Exporta resultados para arquivo texto
    with open(f"{output_dir}/{model_name}_ai_validation_metrics.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Renderiza gráficos comparativos de dispersão e histogramas de erro
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
        ax_scat.set_xlabel("Medição RSRP real (dBm)", fontsize=20)
        ax_scat.set_ylabel("Predição RSRP (dBm)", fontsize=20)
        ax_scat.grid(True, linestyle='--', alpha=0.5)

        ax_hist.hist(err, bins=50, color='orange', edgecolor='black', alpha=0.7)
        ax_hist.axvline(x=0, color='red', linestyle='dashed', linewidth=2)
        ax_hist.set_title("Distribuição do Erro Residual", fontsize=16)
        ax_hist.set_xlabel("Erro (Predito - Real) [dB]", fontsize=20)
        ax_hist.set_ylabel("Frequência", fontsize=20)
        ax_hist.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{output_dir}/{model_name}_ai_validation_plots.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Gera gráfico da Função de Distribuição Acumulada (CDF) do erro absoluto
    plt.figure(figsize=(10, 7))
    for err, label, color in zip([raw_errors, calib_errors], ['Bruto', 'Calibrado'], ['gray', 'green']):
        abs_err = np.abs(err)
        sorted_err = np.sort(abs_err)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        plt.plot(sorted_err, cdf, label=label, color=color, linewidth=2)
        
        # Identifica o valor do erro no percentil 90
        p90_idx = np.where(cdf >= 0.90)[0][0]
        p90_val = sorted_err[p90_idx]
        
        # Adiciona linha vertical e anotação no eixo X para o P90
        plt.axvline(x=p90_val, color=color, linestyle='--', alpha=0.5)
        plt.text(p90_val, 0.05, f"P90: {p90_val:.1f}dB", color=color, rotation=90, verticalalignment='bottom', fontweight='bold')

    plt.axhline(y=0.90, color='red', linestyle='dotted', alpha=0.7, label='Percentil 90')
    plt.title(f"CDF do Erro Absoluto - {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Erro Absoluto |dB|", fontsize=12)
    plt.ylabel("Probabilidade Acumulada", fontsize=12)
    plt.xlim([0, 50])
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize='large')
    plt.savefig(f"{output_dir}/{model_name}_ai_validation_cdf.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Gera visualização 3D correlacionando erro com características físicas do ambiente
    fig3d = plt.figure(figsize=(24, 8))
    ax1 = fig3d.add_subplot(131, projection='3d')
    sc3d = ax1.scatter(strict_gdf['Elevation'], strict_gdf['Clutter'], calib_errors, c=calib_errors, cmap='coolwarm', vmin=-30, vmax=30, alpha=0.6)
    ax1.set_xlabel('Elevação (m)'); ax1.set_ylabel('Classe Terreno'); ax1.set_zlabel('Erro Calibrado (dB)')
    ax1.set_title('Espalhamento 3D: Erro vs Ambiente', fontweight='bold')
    fig3d.colorbar(sc3d, ax=ax1, pad=0.1, fraction=0.02).set_label('Erro Ocorrido (dB)')

    ax2 = fig3d.add_subplot(132)
    ax2.scatter(strict_gdf['Elevation'], calib_errors, alpha=0.3, color='purple', s=10)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Elevação (m)'); ax2.set_ylabel('Erro (dB)')
    ax2.set_title('Erro por Altitude', fontweight='bold'); ax2.grid(True, linestyle='--', alpha=0.5)

    ax3 = fig3d.add_subplot(133)
    ax3.scatter(strict_gdf['Clutter'], calib_errors, alpha=0.3, color='teal', s=10)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    
    # Mapeia códigos de Clutter para nomes amigáveis no gráfico
    dicionario_clutter = {
        0.0: "Sem dados", 1.0: "Urbana Muito Alta", 2.0: "Urbana Alta", 3.0: "Urbana Média", 
        4.0: "Urbana Baixa", 5.0: "Suburbana", 6.0: "Suburbana Desordenada", 
        7.0: "Área Urbana Arborizada", 8.0: "Área Urbana Aberta", 9.0: "Área Industrial Baixa", 
        10.0: "Área Industrial Alta", 11.0: "Área Aberta", 12.0: "Aeroporto Terminal", 
        13.0: "Aeroporto Pista", 14.0: "Vegetação de Alto Porte", 15.0: "Vegetação de Médio Porte", 
        16.0: "Vegetação de Baixo Porte", 17.0: "Pastagem", 18.0: "Agrícolas", 
        19.0: "Área de Reflorestamento", 20.0: "Oceano", 21.0: "Água", 22.0: "Dunas", 
        23.0: "Afloramento Rochoso", 24.0: "Mangue", 25.0: "Porto", 26.0: "Logradouro", 
        27.0: "Rodovias", 28.0: "Ferrovias"
    }
    unique_clutters = sorted(strict_gdf['Clutter'].dropna().unique())
    labels = [dicionario_clutter.get(val, str(val)) for val in unique_clutters]
    ax3.set_xticks(unique_clutters)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.set_xlabel('Classe de uso do solo'); ax3.set_ylabel('Erro (dB)')
    ax3.set_title('Erro por Categoria de Clutter', fontweight='bold'); ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_ai_validation_env_merit.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig3d)
    
    print(f"Validação espacial concluída em {(time.time() - start_time) / 60.0:.2f} minutos")

def main():
    '''
    Loop principal para validar sequencialmente os modelos XGB, MLP e Híbrido.
    '''
    models = ["dt", "xgb", "mlp", "hybrid"]
    for model in models:
        parquet_file = os.path.join(repo_root, "data", "luiz_costa", "inference_results", f"ai_coverage_30m_{model}.parquet")
        output_dir = os.path.join(repo_root, "data", "luiz_costa", f"spatial_validation_{model}")
        if os.path.exists(parquet_file):
            validate_ai_coverage(parquet_file, DT_PATH, output_dir, model)
        else:
            print(f"Resultados Parquet não encontrados para o modelo {model}")

if __name__ == "__main__":
    main()