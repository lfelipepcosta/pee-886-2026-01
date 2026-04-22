import pandas as pd
import numpy as np
import math
import os
from qml.luiz_costa.loaders.data_loader import DataLoader5G

class InferenceGridLoader:
    def __init__(self, base_loader: DataLoader5G, output_dir="/home/luiz.costa/anatel_data/qml_data"):
        self.base_loader = base_loader
        self.output_dir = output_dir
        self.grid_path = os.path.join(self.output_dir, "grid_inference_features_30m.csv")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_or_generate_grid(self):
        if os.path.exists(self.grid_path):
            print(f"Grade de inferência encontrada no cache: {self.grid_path}")
            return pd.read_csv(self.grid_path, low_memory=False)

        print("Iniciando geração da malha mestre de 30m (Isso pode demorar um pouco)...")
        if self.base_loader.df_licensing is None:
            self.base_loader.df_licensing = pd.read_csv(self.base_loader.path_licensing, low_memory=False)

        df_dt = pd.read_csv(self.base_loader.path_5g, low_memory=False)
        min_lat, max_lat = df_dt['Latitude'].min(), df_dt['Latitude'].max()
        min_lon, max_lon = df_dt['Longitude'].min(), df_dt['Longitude'].max()
        
        buffer_deg = 1.0 / 111.32
        min_lat -= buffer_deg; max_lat += buffer_deg
        min_lon -= buffer_deg; max_lon += buffer_deg
        
        res_m = 30.0
        lat_step = (res_m / 1000.0) / 111.32
        cos_lat = math.cos(math.radians((min_lat + max_lat) / 2))
        lon_step = (res_m / 1000.0) / (111.32 * cos_lat)
        
        lats = np.arange(math.floor(min_lat / lat_step) * lat_step, math.ceil(max_lat / lat_step) * lat_step, lat_step)
        lons = np.arange(math.floor(min_lon / lon_step) * lon_step, math.ceil(max_lon / lon_step) * lon_step, lon_step)
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        
        grid_df = pd.DataFrame({'Latitude': lat_mesh.flatten().round(6), 'Longitude': lon_mesh.flatten().round(6)})
        print(f"Topologia inicial gerada com {len(grid_df):,} pixels.")
        
        grid_df['Freq_Medida_DT_MHz'] = -1
        grid_df = self.base_loader._extract_raster_values(grid_df, self.base_loader.path_height, 'Height_m')
        grid_df = self.base_loader._extract_raster_values(grid_df, self.base_loader.path_clutter, 'Clutter_Class')
        grid_df = self.base_loader._find_nearest_antenna(grid_df, tech='NR|5G|IMT')
        
        freq_raw = grid_df['Antena_FreqTx'].astype(float)
        grid_df['Freq_Medida_DT_MHz'] = np.where(freq_raw > 100000, freq_raw / 1000.0, freq_raw)
        
        features_required = [
            'Dist_Antena_m', 'Delta_Azimute', 'Height_m', 'Antena_Altura', 
            'Antena_Ganho', 'Antena_AnguloMeiaPotencia', 'Antena_FrenteCosta',
            'Antena_AnguloElevacao', 'Freq_Medida_DT_MHz', 'Clutter_Class'
        ]
        
        final_df = grid_df[['Latitude', 'Longitude'] + features_required].copy()
        final_df['Clutter_Class'] = final_df['Clutter_Class'].fillna(0).astype(int).astype(str)
        final_df = final_df.dropna()
        
        print(f"Salvando arquivo de grade consolidada em: {self.grid_path}")
        final_df.to_csv(self.grid_path, index=False)
        return final_df