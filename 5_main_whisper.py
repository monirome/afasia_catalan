import yaml
import json
import time
import pandas as pd
from logger_config import setup_logger
from optuna_function import objective
from model_config import initialize_model
from dataset_preparation import load_and_prepare_dataset
from training_setup import train_model, compute_metrics, DataCollatorSpeechSeq2SeqWithPadding
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import wandb
import os

# Definir la ruta donde se guardará el archivo Excel
EXCEL_FILE_PATH = '/home/u917/PROJECT/aphasia/training_logs/training_log.xlsx'

# Crear el directorio si no existe
os.makedirs(os.path.dirname(EXCEL_FILE_PATH), exist_ok=True)

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_to_excel(data, filename):
    df = pd.DataFrame([data])
    if os.path.exists(filename):
        df_existing = pd.read_excel(filename)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_excel(filename, index=False)

def main():
    logger = setup_logger()
    config = load_config('config.yaml')

    # Imprimir la configuración al inicio del script
    logger.info("Configuración del entrenamiento:")
    logger.info(yaml.dump(config, default_flow_style=False))

    if not torch.cuda.is_available():
        logger.error("CUDA no está disponible. Asegúrate de que el entorno esté correctamente configurado.")
        return

    if 'learning_rate' in config['training_args']:
        config['training_args']['learning_rate'] = float(config['training_args']['learning_rate'])

    if config.get('enable_wandb', False):
        wandb.login(key=config['wandb']['key'])
        wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'])
    else:
        config['training_args']['report_to'] = []

    slurm_job_id = os.getenv('SLURM_JOB_ID', 'N/A')
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    excel_data = {
        'slurm_job_id': slurm_job_id,
        'date': current_time,
        'hyperparameters': json.dumps(config['training_args'])
    }

    if config.get('enable_optuna', False):
        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, config), n_trials=10)

        best_hyperparameters = study.best_trial.params
        logger.info(f"Best hyperparameters: {best_hyperparameters}")

        with open("/home/u917/PROJECT/aphasia/best_hyperparameters_whisper.json", "w") as hp_file:
            json.dump(best_hyperparameters, hp_file)

        logger.info("Best hyperparameters saved in best_hyperparameters.json")
        excel_data['best_hyperparameters'] = json.dumps(best_hyperparameters)
    else:
        logger.info("Optuna is disabled. Running with default configuration.")
        try:
            metrics = train_model(config, logger)
            for key, value in metrics.items():
                excel_data[key] = value
        except RuntimeError as e:
            logger.error(f"Runtime error during training: {e}")
            torch.cuda.empty_cache()

    save_to_excel(excel_data, EXCEL_FILE_PATH)

if __name__ == "__main__":
    main()
