import pandas as pd
import rasterio
import numpy as np
import os
from tqdm import tqdm
from sklearn.neighbors import BallTree

def convert_arfcn_5g(n_ref):
    try:
        n_ref = int(float(n_ref))
        if 0 <= n_ref <= 599999: return n_ref * 0.005
        elif 600000 <= n_ref <= 2016667: return 3000.0 + (n_ref - 600000) * 0.015
        elif 2016668 <= n_ref <= 3279165: return 24250.0 + (n_ref - 2016667) * 0.060
    except: pass
    return np.nan

class DataLoader5G:
    def __init__(self, base_path="/home/luiz.costa/anatel_data/_mapas", cache_dir="/home/luiz.costa/anatel_data/qml_data"):
        self.base_path = base_path
        self.cache_dir = cache_dir
        self.path_licensing = os.path.join(base_path, "Dados_DT_Mapa", "csv_licenciamento_RJ.csv")
        self.path_5g = os.path.join(base_path, "Dados_DT", "DT_Ntero_5G.csv")
        self.path_clutter = os.path.join(base_path, "RJ", "RJ", "RJ_Clutter_v5.tif")
        self.path_height = os.path.join(base_path, "RJ", "RJ", "RJ_Heights.tif")
        
        self.df_licensing = None
        os.makedirs(self.cache_dir, exist_ok=True)

    def _extract_raster_values(self, df, raster_path, col_name):
        if not os.path.exists(raster_path):
            df[col_name] = np.nan
            return df

        mask_valid = df['Latitude'].notna() & df['Longitude'].notna()
        df_valid = df[mask_valid]
        coords = [(lon, lat) for lon, lat in zip(df_valid['Longitude'], df_valid['Latitude'])]
        extracted_values = []
        
        try:
            with rasterio.open(raster_path) as src:
                nodata = src.nodata
                print(f"Extraindo mapas para {col_name}...")
                for val in tqdm(src.sample(coords), total=len(coords)):
                    v = val[0]
                    if nodata is not None and v == nodata: extracted_values.append(np.nan)
                    elif v < -10000: extracted_values.append(np.nan)
                    else: extracted_values.append(v)
            df[col_name] = np.nan
            df.loc[mask_valid, col_name] = extracted_values
        except Exception as e:
            print(f"Erro ao extrair {col_name}: {e}")
            df[col_name] = np.nan
        return df

    def _find_nearest_antenna(self, df_dt, tech):
        print(f"Buscando antenas {tech} mais próximas...")
        mask_tech = self.df_licensing['Tecnologia'].astype(str).str.contains(tech, case=False, na=False)
        df_lic_filtered = self.df_licensing[mask_tech].copy().dropna(subset=['Latitude', 'Longitude'])
        
        if df_lic_filtered.empty: return df_dt

        dt_coords_rad = np.radians(df_dt[['Latitude', 'Longitude']].fillna(0).values)
        lic_coords_rad = np.radians(df_lic_filtered[['Latitude', 'Longitude']].values)

        k_neighbors = 15
        tree = BallTree(lic_coords_rad, metric='haversine')
        distances, indices = tree.query(dt_coords_rad, k=k_neighbors)

        n_points = len(df_dt)
        best_antenna_idx = np.zeros(n_points, dtype=int)
        best_score = np.full(n_points, -np.inf)

        lat2, lon2 = dt_coords_rad[:, 0], dt_coords_rad[:, 1]
        dt_freq = df_dt['Freq_Medida_DT_MHz'].fillna(-1).values
        
        for c in range(k_neighbors):
            cand_indices = indices[:, c]
            cand_dist = distances[:, c] * 6371000  

            cand_lat = np.radians(df_lic_filtered['Latitude'].values[cand_indices])
            cand_lon = np.radians(df_lic_filtered['Longitude'].values[cand_indices])
            cand_az = np.nan_to_num(pd.to_numeric(df_lic_filtered['Azimute'].values[cand_indices], errors='coerce'), nan=0.0)
            cand_freq = pd.to_numeric(df_lic_filtered['FreqTxMHz'].values[cand_indices], errors='coerce')
            cand_beam = pd.to_numeric(df_lic_filtered['AnguloMeiaPotenciaAntena'].values[cand_indices], errors='coerce')
            cand_fc = pd.to_numeric(df_lic_filtered['FrenteCostaAntena'].values[cand_indices], errors='coerce')
            cand_gain = np.nan_to_num(pd.to_numeric(df_lic_filtered['GanhoAntena'].values[cand_indices], errors='coerce'), nan=15.0)

            freq_valid = (np.abs(cand_freq - dt_freq) <= 50) | (dt_freq == -1)

            dlon = lon2 - cand_lon
            y = np.sin(dlon) * np.cos(lat2)
            x = np.cos(cand_lat) * np.sin(lat2) - np.sin(cand_lat) * np.cos(lat2) * np.cos(dlon)
            
            bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
            delta_az = np.minimum(np.abs(cand_az - bearing), 360 - np.abs(cand_az - bearing))

            beamwidth = np.where(cand_beam > 0, cand_beam, 65.0) 
            front_back_ratio = np.where(cand_fc > 0, cand_fc, 25.0)  
            
            azimuth_attenuation = np.minimum(12 * (delta_az / beamwidth)**2, front_back_ratio)
            path_loss = 20 * np.log10(cand_dist + 1)
            
            estimated_signal = np.where(freq_valid, cand_gain - azimuth_attenuation - path_loss, -9999)

            better_mask = estimated_signal > best_score
            best_score[better_mask] = estimated_signal[better_mask]
            best_antenna_idx[better_mask] = c

        final_indices = indices[np.arange(n_points), best_antenna_idx]
        found_antennas = df_lic_filtered.iloc[final_indices].reset_index(drop=True)
        
        df_dt['Dist_Antena_m'] = distances[np.arange(n_points), best_antenna_idx] * 6371000
        df_dt['Antena_Ganho'] = pd.to_numeric(found_antennas['GanhoAntena'].values, errors='coerce')
        df_dt['Antena_Altura'] = found_antennas['AlturaAntena'].values
        df_dt['Antena_AnguloMeiaPotencia'] = pd.to_numeric(found_antennas['AnguloMeiaPotenciaAntena'].values, errors='coerce')
        df_dt['Antena_FrenteCosta'] = pd.to_numeric(found_antennas['FrenteCostaAntena'].values, errors='coerce')
        df_dt['Antena_AnguloElevacao'] = pd.to_numeric(found_antennas['AnguloElevacao'].values, errors='coerce')

        cand_az = np.nan_to_num(pd.to_numeric(found_antennas['Azimute'].values, errors='coerce'), nan=0.0)
        cand_lat, cand_lon = np.radians(found_antennas['Latitude'].values), np.radians(found_antennas['Longitude'].values)
        dlon = lon2 - cand_lon
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(cand_lat) * np.sin(lat2) - np.sin(cand_lat) * np.cos(lat2) * np.cos(dlon)
        bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
        delta_az = np.abs(cand_az - bearing)
        df_dt['Delta_Azimute'] = np.minimum(delta_az, 360 - delta_az)
        
        return df_dt

    def load_all_datasets(self):
        cache_file = os.path.join(self.cache_dir, "dt_5g_final_processed.csv")
        
        if os.path.exists(cache_file):
            print(f"Lendo dados consolidados do cache: {cache_file}")
            return pd.read_csv(cache_file, low_memory=False)
            
        print("Montando dataset a partir do zero...")
        self.df_licensing = pd.read_csv(self.path_licensing, low_memory=False)
        df_5g = pd.read_csv(self.path_5g, low_memory=False)
        
        df_5g = self._extract_raster_values(df_5g, self.path_height, 'Height_m')
        df_5g = self._extract_raster_values(df_5g, self.path_clutter, 'Clutter_Class')
        
        df_5g.columns = df_5g.columns.str.strip()
        if 'SSB NR-ARFCN' in df_5g.columns:
            df_5g['Freq_Medida_DT_MHz'] = df_5g['SSB NR-ARFCN'].apply(convert_arfcn_5g)
            
        df_5g = self._find_nearest_antenna(df_5g, tech='NR|5G|IMT')
        
        df_5g.to_csv(cache_file, index=False)
        print("Dataset criado e armazenado no cache.")
        return df_5g