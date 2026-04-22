import pandas as pd
import joblib
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(repo_root)

from qml.luiz_costa.loaders.data_loader import DataLoader5G
from qml.luiz_costa.loaders.grid_loader import InferenceGridLoader
from qml.luiz_costa.visualization.plotting import plot_coverage_map

MODEL_PATH = os.path.join(repo_root, "data", "luiz_costa", "trained_models", "best_pipeline_hybrid.joblib")
OUTPUT_DIR = os.path.join(repo_root, "data", "luiz_costa", "inference_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Inferência e Mapeamento Espacial (Híbrido Quântico)")

    base_loader = DataLoader5G()
    grid_loader = InferenceGridLoader(base_loader)
    df_grid = grid_loader.load_or_generate_grid()

    features = [
        'Dist_Antena_m', 'Delta_Azimute', 'Height_m', 'Antena_Altura', 
        'Antena_Ganho', 'Antena_AnguloMeiaPotencia', 'Antena_FrenteCosta',
        'Antena_AnguloElevacao', 'Freq_Medida_DT_MHz', 'Clutter_Class'
    ]

    pipeline = joblib.load(MODEL_PATH)
    df_grid['RSRP_dBm'] = pipeline.predict(df_grid[features]).astype('float32')
    
    df_route = pd.read_csv(base_loader.path_5g, usecols=['Latitude', 'Longitude']).dropna()
    df_antennas = df_grid[['Antena_Lat', 'Antena_Lon']].drop_duplicates() if 'Antena_Lat' in df_grid.columns else pd.DataFrame()
    
    plot_coverage_map(df_grid, df_route, df_antennas, model_name="Hybrid_Quantum", target_name="SS-RSRP", output_dir=OUTPUT_DIR)

    output_parquet = os.path.join(OUTPUT_DIR, "ai_coverage_30m_hybrid.parquet")
    df_grid[['Latitude', 'Longitude', 'RSRP_dBm']].to_parquet(output_parquet, index=False)

if __name__ == "__main__":
    main()