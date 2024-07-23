import pandas as pd
from datasets import Dataset, Audio
import numpy as np

def load_and_prepare_dataset(metadata_path: str, feature_extractor, tokenizer, sampling_rate: int = 16000):
    metadata_df = pd.read_csv(metadata_path)

    if metadata_df.empty:
        raise ValueError("el archivo de metadatos esta vacio")

    required_columns = ['name_chunk_audio', 'Marca', 'WAB_AQ']
    for column in required_columns:
        if (column not in metadata_df.columns):
            raise ValueError(f"La columna requerida '{column}' no está presente en el archivo de metadatos.")
        
    metadata_df['name_chunk_audio'] = metadata_df['name_chunk_audio'].astype(str)
    metadata_df = metadata_df.dropna(subset=['Marca', 'WAB_AQ'])
    metadata_df['Marca'] = metadata_df['Marca'].apply(lambda x: str(x) if not pd.isna(x) else None)

    dataset = Dataset.from_dict({
        'audio': metadata_df['name_chunk_audio'].tolist(),
        'sentence': metadata_df['Marca'].tolist(),
        'WAB_AQ': metadata_df['WAB_AQ'].tolist()
    })
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    def prepare_dataset(batch):
        try:
            # load and resample audio data from 48 to 16kHz
            audio = batch["audio"]
            sampling_rate = batch['audio']['sampling_rate']
            # compute log-Mel input features from input audio array
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=sampling_rate).input_features[0]

            # encode target text to label ids
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
        except Exception as e:
            print(f"Error procesando el archivo {batch['audio']['path']}: {e}")
            batch["input_features"] = None
            batch["labels"] = None
        return batch

    prepared_dataset = dataset.map(prepare_dataset)

    if len(prepared_dataset) == 0:
        raise ValueError("El conjunto de datos preparado está vacío.")

    return prepared_dataset.train_test_split(test_size=0.1)
