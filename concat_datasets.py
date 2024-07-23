#hago esto primero python3 -m pip install --user pandas
# python3 -m pip install --user librosa

import pandas as pd
import ast

df_es = pd.read_csv('/home/u917/PROJECT/aphasia/data_es/df_ES_transcripciones_chunk_expandido.csv')
df_catalan = pd.read_csv('/home/u917/PROJECT/aphasia/data_catalan/df_catalan_transcrip_chunk_info_actualizado_expandido.csv')

df_es.rename(columns={
    'mark_start': 'Inicio',
    'mark_end': 'Fin',
    'transcriptions': 'Marca',
    'file_cut': 'name_chunk_audio'
}, inplace=True)

df_es = df_es[[ 'Marca', 'name_chunk_audio', 'WAB_AQ']] #'Inicio', 'Fin', 'sex', 'age'

df_catalan.loc[df_catalan['Transcrip_name'] == '001_003_CAT_Llenguatge_espontani', 'QA'] = 34.7
df_catalan.rename(columns={
    'NumId': 'ID',
    'QA': 'WAB_AQ',
    'Gènere': 'sex',
    'Edat': 'age'
}, inplace=True)

df_catalan = df_catalan[['Marca', 'name_chunk_audio', 'WAB_AQ']] #'Inicio', 'Fin', 'sex', 'age'

df_concatenado = pd.concat([df_es, df_catalan], ignore_index=True)

#funcion para clasificar el grado de afasia
def categorizar_aphasia(score):
    if score >= 76:
        return 'Leve'
    elif score >= 51:
        return 'Moderada'
    elif score >= 26:
        return 'Severa'
    else:
        return 'Muy Severa'

df_concatenado['aphasia_category'] = df_concatenado['WAB_AQ'].apply(categorizar_aphasia)

df_concatenado['WAB_AQ_round'] = df_concatenado['WAB_AQ'].round().astype(int)

output_path = '/home/u917/PROJECT/aphasia/data_concat/concat_dataset.csv'
df_concatenado.to_csv(output_path, index=False)

print("El DataFrame se ha expandido y guardado correctamente.")