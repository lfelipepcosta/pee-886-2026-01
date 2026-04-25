import pandas as pd
import numpy as np
import math
import os
from qml.luiz_costa.loaders.data_loader import DataLoader5G

class InferenceGridLoader:
    '''
    Classe para gerar uma grade de pontos para inferência e mapeamento.
    Cria pontos espaçados regularmente dentro da área coberta pelo Drive Test.
    '''
    def __init__(self, base_loader: DataLoader5G, output_dir="/home/luiz.costa/anatel_data/qml_data"):
        '''
        Inicializa o carregador usando um DataLoader5G base e define o caminho de saída.
        '''
        self.base_loader = base_loader
        self.output_dir = output_dir
        self.grid_path = os.path.join(self.output_dir, "grid_inference_features_30m.csv")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_or_generate_grid(self):
        '''
        Recupera a grade de inferência do cache ou gera uma nova se necessário.
        '''
        # Verifica se o arquivo da grade já existe no diretório de cache
        if os.path.exists(self.grid_path):
            print(f"Grade de inferência encontrada no cache em {self.grid_path}")
            return pd.read_csv(self.grid_path, low_memory=False)

        print("Iniciando geração da malha mestre de 30m (isso pode demorar um pouco)")
        # Garante que os dados de licenciamento da Anatel estão carregados
        if self.base_loader.df_licensing is None:
            self.base_loader.df_licensing = pd.read_csv(self.base_loader.path_licensing, low_memory=False)

        # Determina os limites geográficos baseados nos dados reais do Drive Test
        df_dt = pd.read_csv(self.base_loader.path_5g, low_memory=False)
        min_lat, max_lat = df_dt['Latitude'].min(), df_dt['Latitude'].max()
        min_lon, max_lon = df_dt['Longitude'].min(), df_dt['Longitude'].max()
        
        # Adiciona uma margem de segurança de aproximadamente 1km nas bordas
        buffer_deg = 1.0 / 111.32
        min_lat -= buffer_deg; max_lat += buffer_deg
        min_lon -= buffer_deg; max_lon += buffer_deg
        
        # Calcula os passos em graus para atingir a resolução de 30 metros desejada
        res_m = 30.0
        lat_step = (res_m / 1000.0) / 111.32
        cos_lat = math.cos(math.radians((min_lat + max_lat) / 2))
        lon_step = (res_m / 1000.0) / (111.32 * cos_lat)
        
        # Gera o grid de coordenadas usando arange e meshgrid
        lats = np.arange(math.floor(min_lat / lat_step) * lat_step, math.ceil(max_lat / lat_step) * lat_step, lat_step)
        lons = np.arange(math.floor(min_lon / lon_step) * lon_step, math.ceil(max_lon / lon_step) * lon_step, lon_step)
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        
        grid_df = pd.DataFrame({'Latitude': lat_mesh.flatten().round(6), 'Longitude': lon_mesh.flatten().round(6)})
        print(f"Topologia inicial gerada com {len(grid_df):,} pixels")
        
        # Atribui valores de relevo e Clutter para cada ponto da grade gerada
        grid_df['Freq_Medida_DT_MHz'] = -1
        grid_df = self.base_loader._extract_raster_values(grid_df, self.base_loader.path_height, 'Height_m')
        grid_df = self.base_loader._extract_raster_values(grid_df, self.base_loader.path_clutter, 'Clutter_Class')
        # Associa cada ponto à antena de tecnologia 5G mais próxima
        grid_df = self.base_loader._find_nearest_antenna(grid_df, tech='NR|5G|IMT')
        
        # Ajusta a frequência se estiver em kHz para manter o padrão MHz
        freq_raw = grid_df['Antena_FreqTx'].astype(float)
        grid_df['Freq_Medida_DT_MHz'] = np.where(freq_raw > 100000, freq_raw / 1000.0, freq_raw)
        
        # Lista as colunas obrigatórias para a entrada dos modelos de ML
        features_required = [
            'Dist_Antena_m', 'Delta_Azimute', 'Height_m', 'Antena_Altura', 
            'Antena_Ganho', 'Antena_AnguloMeiaPotencia', 'Antena_FrenteCosta',
            'Antena_AnguloElevacao', 'Freq_Medida_DT_MHz', 'Clutter_Class'
        ]
        
        # Finaliza o DataFrame, remove valores nulos e salva CSV no cache
        cols_to_keep = ['Latitude', 'Longitude'] + features_required
        for extra_col in ['Antena_Lat', 'Antena_Lon']:
            if extra_col in grid_df.columns:
                cols_to_keep.append(extra_col)
        
        final_df = grid_df[cols_to_keep].copy()
        final_df['Clutter_Class'] = final_df['Clutter_Class'].fillna(0.0).astype(float)
        final_df = final_df.dropna()
        
        print(f"Salvando o arquivo da grade consolidada em {self.grid_path}")
        final_df.to_csv(self.grid_path, index=False)
        return final_df