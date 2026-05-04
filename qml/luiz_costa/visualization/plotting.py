import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.inspection import permutation_importance

def plot_feature_importance(pipeline, X_val, y_val, target_name, model_name, output_dir):
    '''
    Gera um gráfico analítico mostrando o peso de cada variável no modelo.
    Mostra o quanto o erro aumenta ao embaralhar aleatoriamente cada coluna.
    '''
    print(f"Gerando gráfico de importância por permutação para {model_name}")
    result = permutation_importance(pipeline, X_val, y_val, n_repeats=3, random_state=42, n_jobs=1, scoring='neg_root_mean_squared_error')
    
    # Cria gráfico de barras horizontais ordenado pelas features mais importantes
    plt.figure(figsize=(12, 8))
    importances = pd.Series(result.importances_mean, index=X_val.columns)
    importances.nlargest(15).sort_values().plot(kind='barh', color='navy')
    plt.title(f'Importância Relativa ({model_name}) - {target_name}')
    plt.xlabel('Custo do Erro (RMSE)')
    plt.ylabel('Features')
    plt.tight_layout()
    
    # Salva em formato PDF vetorizado para incluir no documento final
    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '')}_feature_importance_{target_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_actual_vs_predicted(y_test, y_pred, target_name, model_name, output_dir):
    '''
    Plota um gráfico de dispersão comparando os valores reais medidos e os preditos pela IA.
    '''
    print(f"Gerando gráfico de Previsão vs Valores Reais para {model_name}")
    plt.figure(figsize=(10, 8))
    
    # Amostra pontos aleatórios para evitar que o gráfico PDF fique muito pesado
    sample_size = min(25000, len(y_test))
    sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
    y_test_vals = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
    
    # Plota a dispersão azul e a linha de referência ideal vermelha (Y=X)
    plt.scatter(y_test_vals, y_pred[sample_idx], alpha=0.3, color='royalblue', s=12)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(f'Propagação Verdadeira x Sinal Previsto ({model_name}) - {target_name}')
    plt.xlabel('Verdadeiro Sinal Medido Efetivo')
    plt.ylabel('Sinal Previsto')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '')}_actual_vs_predicted_{target_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_error_distribution(y_test, y_pred, target_name, model_name, output_dir):
    '''
    Exibe a distribuição estatística dos erros (Resíduos) em um histograma.
    '''
    print(f"Gerando gráfico de distribuição de resíduos para {model_name}")
    errors = y_test - y_pred

    plt.figure(figsize=(10, 6))
    
    # Plota o histograma com a curva KDE e marca o erro zero com linha tracejada
    sns.histplot(errors, bins=50, kde=True, color='purple')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
    plt.title(f'Distribuição de Erro ({model_name}) - {target_name}')
    plt.xlabel('Erro Residual (dB)')
    plt.ylabel('Frequência')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '')}_error_distribution_{target_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_coverage_map(df_coverage, df_route, df_antennas, model_name, target_name, output_dir):
    '''
    Renderiza um mapa de calor geográfico comparando predição de cobertura e rota drive test.
    Salva em formato PDF usando o sistema de coordenadas geográficas SIRGAS 2000.
    '''
    print(f"Renderizando mapa de cobertura para o modelo {model_name}")
    
    # Cria o GeoDataFrame para plotagem espacial SIRGAS 2000 (EPSG:4674)
    coverage_gdf = gpd.GeoDataFrame(
        df_coverage, 
        geometry=[Point(xy) for xy in zip(df_coverage['Longitude'], df_coverage['Latitude'])], 
        crs="EPSG:4674"
    )

    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define limites de escala de cor entre o percentil 5 e 95 para ignorar outliers
    vmin_data = coverage_gdf['RSRP_dBm'].quantile(0.05)
    vmax_data = coverage_gdf['RSRP_dBm'].quantile(0.95)

    coverage_gdf.plot(
        column='RSRP_dBm', ax=ax, cmap='RdYlGn', vmin=vmin_data, vmax=vmax_data, 
        legend=True, markersize=2, alpha=0.8, legend_kwds={'label': 'Predição RSRP (dBm)'}
    )
    
    # Desenha os pontos reais percorridos durante a coleta no Drive Test
    if not df_route.empty:
        route_gdf = gpd.GeoDataFrame(df_route, geometry=[Point(xy) for xy in zip(df_route['Longitude'], df_route['Latitude'])], crs="EPSG:4674")
        route_gdf.plot(ax=ax, color='blue', markersize=0.5, alpha=0.4, label="Rota Drive Test Real")

    # Marca a posição exata das antenas de 5G selecionadas
    if not df_antennas.empty:
        antennas_gdf = gpd.GeoDataFrame(df_antennas, geometry=[Point(xy) for xy in zip(df_antennas['Antena_Lon'], df_antennas['Antena_Lat'])], crs="EPSG:4674")
        antennas_gdf.plot(ax=ax, color='black', marker='x', markersize=80, linewidths=1.5, label="Sites (Antenas)", zorder=10)

    # Força os limites do mapa a casarem com a área predita
    min_lon, max_lon = df_coverage['Longitude'].min(), df_coverage['Longitude'].max()
    min_lat, max_lat = df_coverage['Latitude'].min(), df_coverage['Latitude'].max()
    ax.set_xlim([min_lon, max_lon])
    ax.set_ylim([min_lat, max_lat])

    # Configurações finais de título, legenda e grade
    plt.title(f"Mapa de Cobertura Espacial ({model_name}) - {target_name}", fontsize=16)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '')}_coverage_map_{target_name}.png")
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_learning_curve(history, model_name, target_name, output_dir):
    '''
    Gera um gráfico da curva de aprendizado para monitorar convergência.
    Padroniza todos os erros para RMSE.
    '''
    print(f"Gerando curva de aprendizado para {model_name}")
    plt.figure(figsize=(10, 6))
    
    # Se for XGBoost, o X é estimadores e o histórico já é RMSE
    if model_name.lower() == "xgboost":
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        x_label = 'Estimadores (Árvores)'
    # Se for PyTorch (MLP/Hybrid), o X é épocas e o histórico é MSE (precisa tirar a raiz)
    else:
        train_loss = np.sqrt(history['train_loss'])
        val_loss = np.sqrt(history['val_loss'])
        x_label = 'Épocas de Treinamento'
        
    x_axis = range(1, len(train_loss) + 1)
    
    plt.plot(x_axis, train_loss, 'b-', label='Treino', linewidth=2)
    plt.plot(x_axis, val_loss, 'r-', label='Validação', linewidth=2)
    
    plt.title(f'Curva de Aprendizado - {model_name} ({target_name})', fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Erro (RMSE em dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '')}_learning_curve_{target_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()