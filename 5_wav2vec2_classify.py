import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE  # para balancear

path = '/home/u917/PROJECT/aphasia/classify_codes/'

# Cargar los datasets
train_df = pd.read_csv(path + 'train_df.csv', encoding='utf-8')
test_df = pd.read_csv(path + 'test_df.csv', encoding='utf-8')

# Actualizar los paths a los archivos de audio en el servidor
train_df['name_chunk_audio_path'] = train_df['name_chunk_audio_path'].str.replace(
    '/Users/monicaromero/PycharmProjects/afasia_cat/audios_chunks/',
    '/home/u917/PROJECT/aphasia/data_catalan/audios_chunks/',
    regex=False
)

test_df['name_chunk_audio_path'] = test_df['name_chunk_audio_path'].str.replace(
    '/Users/monicaromero/PycharmProjects/afasia_cat/audios_chunks/',
    '/home/u917/PROJECT/aphasia/data_catalan/audios_chunks/',
    regex=False
)

# Cargar el modelo y el procesador de wav2vec 2.0
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True)

# Función para extraer la capa 6 de Wav2Vec 2.0
def extract_wav2vec2_layer(audio_path, layer_num=6):
    try:
        # Cargar audio y procesarlo con wav2vec2
        y, sr = librosa.load(audio_path, sr=16000)  # Asegúrate de usar 16kHz para wav2vec2
        input_values = processor(y, return_tensors="pt", sampling_rate=16000).input_values

        # Extraer las características de la capa especificada
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.hidden_states  # Lista de capas
            selected_layer = hidden_states[layer_num]  # Capa 6

        # Promediar sobre el tiempo para obtener un solo vector de características
        return torch.mean(selected_layer, dim=1).squeeze().numpy()
    except Exception as e:
        print(f"Error al procesar el archivo {audio_path}: {e}")
        return np.zeros(768)  # Devolver vector de ceros si hay un error (dimensión de capa Wav2Vec2)

# Extraer características con la capa 6 de Wav2Vec 2.0
train_features = []
test_features = []

for path in train_df['name_chunk_audio_path']:
    train_features.append(extract_wav2vec2_layer(path, layer_num=6))  # Extraer la capa 6

for path in test_df['name_chunk_audio_path']:
    test_features.append(extract_wav2vec2_layer(path, layer_num=6))  # Extraer la capa 6

# Convertir las listas de características en DataFrames
train_features_df = pd.DataFrame(train_features)
test_features_df = pd.DataFrame(test_features)

# Preprocesamiento de los datos para LightGBM
# Eliminar columnas innecesarias que son de tipo object y no sirven para el modelo
cols_to_drop = ['CIP', 'Marca', 'Transcrip_name', 'name_chunk_audio', 'balance_group', 'name_chunk_audio_path']
train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

# Convertir la variable objetivo (Fluente/No Fluente) a formato numérico
le = LabelEncoder()
train_df['Fluente/No Fluente'] = le.fit_transform(train_df['Fluente/No Fluente'])
test_df['Fluente/No Fluente'] = le.transform(test_df['Fluente/No Fluente'])

# Concatenar las características extraídas de los audios con el resto del dataset
X_train = pd.concat([train_df.drop('Fluente/No Fluente', axis=1), train_features_df], axis=1)
y_train = train_df['Fluente/No Fluente']

X_test = pd.concat([test_df.drop('Fluente/No Fluente', axis=1), test_features_df], axis=1)
y_test = test_df['Fluente/No Fluente']

# Convertir todos los nombres de columnas a strings para evitar problemas con SMOTE
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Balanceo de clases con SMOTE
sm = SMOTE(sampling_strategy='auto', random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Parámetros modificados para LightGBM
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'max_depth': -1,  # Sin restriccion en la profundidad del arbol
    'num_leaves': 50,  # mayor numero de hojas
    'learning_rate': 0.01,  # reducir la tasa de aprendizaje
    'min_data_in_leaf': 10,  # reducir para permitir mas divisiones
    'feature_fraction': 0.8  # probar un subconjunto de características mas pequeño
}

# Entrenar el modelo usando callbacks para early stopping
train_data = lgb.Dataset(X_train_res, label=y_train_res)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

num_round = 500  
callbacks = [lgb.early_stopping(stopping_rounds=20)]  # aumentar el numero de iteraciones antes del early stopping

bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], callbacks=callbacks)

# Evaluación del modelo
y_pred = bst.predict(X_test)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Imprimir el reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_binary, target_names=['No Fluente', 'Fluente']))

# Imprimir la matriz de confusión
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_binary))
