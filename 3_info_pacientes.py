import pandas as pd
import pyreadstat
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

file_path_chunk = '/Users/monicaromero/PycharmProjects/afasia_cat/data_output/dataset_transcripciones_chunk.csv'
# file_path_valencia = '/Users/monicaromero/PycharmProjects/afasia_cat/Estudi Afàsia - còpia.sav'
file_path_valencia = '/Users/monicaromero/PycharmProjects/afasia_cat/Estudi Afàsia València versió2.sav'
file_path_hospitalet = '/Users/monicaromero/PycharmProjects/afasia_cat/Estudi Afàsia dades Hospitalet/Matriu Variables Isabel Estudi Afàsia.sav'

df_chunks = pd.read_csv(file_path_chunk)
df_valencia, meta_valencia = pyreadstat.read_sav(file_path_valencia)
df_hospitalet, meta_hospitalet = pyreadstat.read_sav(file_path_hospitalet)

df_valencia["id_audio"] = "02_00" + df_valencia["NumId"].astype(int).astype(str)

def convert_code(code):
    """
    Convierte un código dado del formato XX.XXX a XX_XXX. Si el código es '54FC',
    se convertirá a '02_011'.
    """
    if code == '54FC':
        return '02_011'
    return code.replace('.', '_')

def convert_code_2(code):
    """
    Convierte un código dado del formato XX-XXX a XX_XXX
    """
    return code.replace('-', '_')

def extract_prefix(text):
    """
    Extrae la parte del texto que cumple con el formato 'XX_XXX' donde 'X' es un dígito.
    """
    match = re.match(r'(\d{2}_\d{3})', text)
    return match.group(0) if match else None

codes = [
    "02.001", "54FC", "02.002", "02.003", "02.004", "02.005",
    "02.006", "02.007", "02.008", "02.009", "02.010"]

df_valencia['CIP'] = df_valencia['CIP'].apply(convert_code)
df_hospitalet['CIP'] = df_hospitalet['CIP'].apply(convert_code_2)
df_chunks['CIP'] = df_chunks['name_chunk_audio'].apply(extract_prefix)
df_chunks['NumId'] = df_chunks['Transcrip_name'].str.extract(r'_(\d{3})_').astype(int)

df_concatenado = pd.concat([df_valencia, df_hospitalet], ignore_index=True)

df_cruzado = pd.merge(df_chunks, df_concatenado[['CIP', 'Gènere', 'TipusAfàsia', 'LLengWAB', 'Edat', 'Grup', 'QA']], on="CIP", how="left")

df_cruzado['Gènere'] = pd.to_numeric(df_cruzado['Gènere'], errors='coerce')
df_cruzado['TipusAfàsia'] = pd.to_numeric(df_cruzado['TipusAfàsia'], errors='coerce')
df_cruzado['LLengWAB'] = pd.to_numeric(df_cruzado['LLengWAB'], errors='coerce')
df_cruzado['Edat'] = pd.to_numeric(df_cruzado['Edat'], errors='coerce')
df_cruzado['Grup'] = pd.to_numeric(df_cruzado['Grup'], errors='coerce')

df_cruzado['Gènere'] = df_cruzado['Gènere'].astype('Int64')
df_cruzado['TipusAfàsia'] = df_cruzado['TipusAfàsia'].astype('Int64')
df_cruzado['LLengWAB'] = df_cruzado['LLengWAB'].astype('Int64')
df_cruzado['Edat'] = df_cruzado['Edat'].astype('Int64')
df_cruzado['Grup'] = df_cruzado['Grup'].astype('Int64')

# mapeo_grup = {1: "bilingue", 2: "monolingue"}
# mapeo_genere = {1: "hombre", 2: "mujer"}
mapeo_tipus_afasia = {
    "motora": 1, "sensorial": 2, "global": 3, "conduccion": 4,
    "anomica": 5, "motora transcortical": 6, "sensorial transcortical": 7
}
mapeo_llengwab = {"Català": 1, "Castellà": 2}

# df_cruzado['Grup'] = df_cruzado['Grup'].map(mapeo_grup)
# df_cruzado['Gènere'] = df_cruzado['Gènere'].map(mapeo_genere)
df_cruzado['TipusAfàsia'] = df_cruzado['TipusAfàsia'].map(lambda x: mapeo_tipus_afasia.get(str(x).lower(), x) if pd.notna(x) else x)
df_cruzado['LLengWAB'] = df_cruzado['LLengWAB'].map(lambda x: mapeo_llengwab.get(str(x), x))

df_cruzado = df_cruzado[df_cruzado['Duración']>=1]
df_cruzado['Marca'] = df_cruzado['Marca'].str.replace("*inintel·ligible*", "IL", regex=False)

ruta_base = '/Users/monicaromero/PycharmProjects/afasia_cat/data_output/'
df_cruzado.to_csv(ruta_base + 'df_transcrip_chunk_info.csv', index=False, encoding='utf-8')

print(df_cruzado)