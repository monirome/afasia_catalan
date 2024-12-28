import subprocess
import pandas as pd

audio_dir = '/Users/monicaromero/PycharmProjects/afasia_cat/Transcripcions_finals'
data_dir = '/Users/monicaromero/PycharmProjects/afasia_cat/data_output/'
output_dir = '/Users/monicaromero/PycharmProjects/afasia_cat/audios_chunks/'
archivo_m4a = f"{audio_dir}/02_003_CAT_Lamina.m4a"
archivo_wav = archivo_m4a.replace('.m4a', '.wav')

command_convert = ['ffmpeg', '-i', archivo_m4a, '-acodec', 'pcm_s16le', '-ar', '44100', archivo_wav]
subprocess.run(command_convert)

df_transcripciones = pd.read_csv(data_dir+"transcripciones_finales.csv", dtype={"Inicio": float, "Fin": float})

nombres_chunks = []
for index, row in df_transcripciones.iterrows():
    nombre_chunk = f"{row['Transcrip_name']}_{format(row['Inicio'], '.5f')}_{format(row['Fin'], '.5f')}.wav"
    nombres_chunks.append(nombre_chunk)

    input_audio_path = f"{audio_dir}/{row['Transcrip_name']}.wav"
    output_audio_path = f"{output_dir}/{row['Transcrip_name']}_{format(row['Inicio'], '.5f')}_{format(row['Fin'], '.5f')}.wav"

    command = [
        'ffmpeg',
        '-i', input_audio_path,
        '-ss', str(row['Inicio']),
        '-to', str(row['Fin']),
        '-c', 'copy',
        output_audio_path
    ]
    subprocess.run(command)

df_transcripciones['name_chunk_audio'] = nombres_chunks
# rutas_completas = ['/home/u917/PROJECT/aphasia/data_catalan/audios_chunks/' + nombre for nombre in nombres_chunks]
rutas_completas = ['/Users/monicaromero/PycharmProjects/afasia_cat/audios_chunks/' + nombre for nombre in nombres_chunks]
df_transcripciones['name_chunk_audio_path'] = rutas_completas
df_transcripciones.to_csv(data_dir + "dataset_transcripciones_chunk.csv", index=False)

print("Corte de audio completado.")

