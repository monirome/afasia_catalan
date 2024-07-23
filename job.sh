#!/bin/bash
#----------------------- Start job description -----------------------
#SBATCH --partition=standard-gpu
#SBATCH --job-name=a100_small
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=10G 
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXX@mail.com
#SBATCH --chdir=/home/u917/PROJECT/aphasia
#SBATCH --output=/home/u917/PROJECT/aphasia/slurm_outputs/slurm-%j.out  # Añadir el ID de trabajo en el nombre del archivo de salida
#------------------------ End job description ------------------------

module purge
module load Python/3.9.5-GCCcore-10.3.0 CUDA/11.3.1

# Crear y activar el entorno virtual
rm -rf /home/u917/PROJECT/aphasia/venv
python3 -m venv /home/u917/PROJECT/aphasia/venv
source /home/u917/PROJECT/aphasia/venv/bin/activate

pip install --no-cache-dir numpy==1.26.4 pandas==2.2.1

pip install -r /home/u917/PROJECT/aphasia/requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers==4.26.0
pip show pyyaml
pip show jiwer
pip install tensorboard
pip install huggingface_hub
pip install wandb
pip install openpyxl
pip install lightgbm
pip install librosa
pip install tabgan==1.3.3


# Exportar variable de entorno para CUDA
export CUDA_LAUNCH_BLOCKING=1

# Exportar clave de API para W&B
export WANDB_API_KEY="50c2664133a05f17cba29eec0576ab3905ed2ec9"

# Crear directorio de salida con timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/home/u917/PROJECT/aphasia/slurm_outputs/whisper_training_output_$TIMESTAMP"
mkdir -p $OUTPUT_DIR

# Ejecutar el script principal
#srun python3 /home/u917/PROJECT/aphasia/5_main_whisper.py --output_dir $OUTPUT_DIR
srun python3 /home/u917/PROJECT/aphasia/aphasia_classifier.py --output_dir $OUTPUT_DIR

# Desactivar el entorno virtual
deactivate
