from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Lista das variáveis numéricas que serão normalizadas
NUMERIC_FEATURES = [
    'Dist_Antena_m', 'Delta_Azimute', 'Height_m', 'Antena_Altura', 
    'Antena_Ganho', 'Antena_AnguloMeiaPotencia', 'Antena_FrenteCosta',
    'Antena_AnguloElevacao', 'Freq_Medida_DT_MHz'
]

# Lista da variável categórica que passará por One-Hot Encoding
CATEGORICAL_FEATURE = ['Clutter_Class']

def create_preprocessor():
    '''
    Cria um objeto ColumnTransformer para processar os dados.
    Aplica StandardScaler em colunas numéricas e OneHotEncoder na categoria Clutter.
    '''
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURE)
        ]
    )