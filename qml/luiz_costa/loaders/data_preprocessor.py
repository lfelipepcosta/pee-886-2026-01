from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NUMERIC_FEATURES = [
    'Dist_Antena_m', 'Delta_Azimute', 'Height_m', 'Antena_Altura', 
    'Antena_Ganho', 'Antena_AnguloMeiaPotencia', 'Antena_FrenteCosta',
    'Antena_AnguloElevacao', 'Freq_Medida_DT_MHz'
]
CATEGORICAL_FEATURE = ['Clutter_Class']

def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURE)
        ]
    )