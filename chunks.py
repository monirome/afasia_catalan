import subprocess
import pandas as pd
import os

audio_dir = '/home/u917/PROJECT/aphasia/data_es/chunks'
output_dir = '/home/u917/PROJECT/aphasia/data_es/chunks'
data_file = '/home/u917/PROJECT/aphasia/data_es/df_ES_clean.csv'
extra_audio_dir = '/home/u917/PROJECT/aphasia/data_catalan/audios_chunks'
data_file_catalan = '/home/u917/PROJECT/aphasia/data_catalan/df_transcrip_chunk_info.csv'

df = pd.read_csv(data_file)
df_catalan = pd.read_csv(data_file_catalan)

def verificar_existencia_archivo(file_path):
    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        return False
    return True

def procesar_audio(row, input_audio_path, output_dir, file_column):
    if not verificar_existencia_archivo(input_audio_path):
        return None

    nombre_base = os.path.splitext(row[file_column])[0]
    extension = os.path.splitext(row[file_column])[1]

    temp_data = []
    for speed, tempo in zip(["original", "slow", "fast"], [1, 0.9, 1.1]):
        output_audio_path = f"{output_dir}/{nombre_base}_{speed}{extension}"
        command = [
            'ffmpeg',
            '-y',
            '-i', input_audio_path,
            '-filter:a', f"atempo={tempo}",
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_audio_path
        ]
        subprocess.run(command)
        temp_data.append({
            **row.to_dict(),
            file_column: output_audio_path,
            'speed': speed
        })

    return temp_data

def expandir_dataframe(df, audio_dir, output_dir, file_column):
    new_rows = []
    for _, row in df.iterrows():
        result = procesar_audio(row, f"{audio_dir}/{row[file_column]}", output_dir, file_column)
        if result:
            new_rows.extend(result)
    return pd.DataFrame(new_rows)

df_expanded = expandir_dataframe(df, audio_dir, output_dir, 'file_cut')
nuevo_nombre_espanol = data_file.replace('df_ES_clean.csv', 'df_ES_transcripciones_chunk_expandido.csv')
df_expanded.to_csv(nuevo_nombre_espanol, index=False)

df_catalan_expanded = expandir_dataframe(df_catalan, extra_audio_dir, extra_audio_dir, 'name_chunk_audio')
nuevo_nombre_catalan = data_file_catalan.replace('df_transcrip_chunk_info.csv', 'df_catalan_transcrip_chunk_info_actualizado_expandido.csv')
df_catalan_expanded.to_csv(nuevo_nombre_catalan, index=False)

