import textgrid
import pandas as pd
import re
import os
import unicodedata


def limpiar_texto(texto):
    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join([c for c in texto if not unicodedata.combining(c)])
    texto = texto.lower()  # convertir a minúsculas
    texto = re.sub(r'\.\s*$', '', texto)  # eliminar punto final
    texto = texto.strip()  # eliminar espacios vacíos al comienzo y final de las frases
    texto = re.sub(r'[,.!¿?¡()"\-:;]', '', texto)   # eliminar comas, exclamación, interrogación...etc
    return texto


textgrid_path = '/Users/monicaromero/PycharmProjects/afasia_cat/Transcripcions_finals'

dataframes_transcrip = []
dataframes_pausa = []
dataframes_coment_extra = []
dataframes_coment = []

for archivo in os.listdir(textgrid_path):
    if archivo.endswith('.TextGrid'):
        ruta_completa = os.path.join(textgrid_path, archivo)
        tg = textgrid.TextGrid.fromFile(ruta_completa)

        tier_names = []
        min_times = []
        max_times = []
        marks = []

        for tier in tg:
            for interval in tier:
                tier_names.append(tier.name)
                min_times.append(interval.minTime)
                max_times.append(interval.maxTime)
                marks.append(interval.mark)

        df = pd.DataFrame({
            'Tier': tier_names,
            'Inicio': min_times,
            'Fin': max_times,
            'Marca': marks
        })

        df['Transcrip_name'] = archivo.replace('.TextGrid', '')

        df['Marca'] = df['Marca'].apply(limpiar_texto)
        df = df[df['Marca'].str.strip().astype(bool)] #quitar vacias
        df_transcrip = df[df['Tier'].str.contains('transcripció', case=False, na=False)]
        df_transcrip = df_transcrip[~df_transcrip['Marca'].str.contains(r'\*interlocutor\*')]
        df_transcrip = df_transcrip[df_transcrip['Marca'] != "xxx"]
        # df_transcrip = df_transcrip[df_transcrip['Marca'].str.strip() != "*inintel·ligible*"]

        df_pausa = df[df['Tier'] == 'Parla-pausa']
        df_coment_extra = df[df['Tier'] == 'Comentaris extra']
        df_coment = df[df['Tier'] == 'Comentaris']

        dataframes_transcrip.append(df_transcrip)
        dataframes_pausa.append(df_pausa)
        dataframes_coment_extra.append(df_coment_extra)
        dataframes_coment.append(df_coment)

df_final_transcrip = pd.concat(dataframes_transcrip, ignore_index=True)
df_final_pausa = pd.concat(dataframes_pausa, ignore_index=True)
df_final_coment_extra = pd.concat(dataframes_coment_extra, ignore_index=True)
df_final_coment = pd.concat(dataframes_coment, ignore_index=True)

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df_final_transcrip)

df_final_transcrip = df_final_transcrip.drop('Tier', axis=1)
df_final_transcrip = df_final_transcrip.drop_duplicates().reset_index(drop=True)

df_final_transcrip['Duración'] = df_final_transcrip['Fin'] - df_final_transcrip['Inicio']
tiempo_total_segundos = df_final_transcrip['Duración'].sum()

tiempo_total_horas = int(tiempo_total_segundos // 3600)
tiempo_total_minutos = int((tiempo_total_segundos % 3600) // 60)
tiempo_total_segundos_resto = tiempo_total_segundos % 60

tiempo_legible = "{:02d} horas {:02d} minutos {:05.2f} segundos".format(
    tiempo_total_horas,
    tiempo_total_minutos,
    tiempo_total_segundos_resto)

print(tiempo_legible)


ruta_base = '/Users/monicaromero/PycharmProjects/afasia_cat/data_output/'
df_final_transcrip.to_csv(ruta_base + 'transcripciones_finales.csv', index=False, encoding='utf-8')
df_final_pausa.to_csv(ruta_base + 'pausas_finales.csv', index=False, encoding='utf-8')
df_final_coment_extra.to_csv(ruta_base + 'comentarios_extra_finales.csv', index=False, encoding='utf-8')
df_final_coment.to_csv(ruta_base + 'comentarios_finales.csv', index=False, encoding='utf-8')

# print(df_final_transcrip[(df_final_transcrip['Transcrip_name'] == "02_007_CAT_Lámina"))
