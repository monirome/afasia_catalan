#!/bin/bash
##----------------------Load apps -------------------------------------
# See all available modules with: "module avail"
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load FFmpeg/6.0-GCCcore-12.3.0
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
module load TensorFlow/2.4.1-fosscuda-2020b
##--------------------Install python libraries-------------------------------------
pip install --no-cache-dir numpy==1.20.3 pandas==1.3.0
pip install pandas pydub
pip install --upgrade pyyaml
pip install accelerate -U
pip install transformers[torch]
pip install SoundFile
pip install soundfile
pip install editdistance
pip install sentencepiece
pip install tensorboardX
pip install tqdm
pip install jiwer
pip install hydra-ax-sweeper #==1.2
pip install librosa
pip install transformers #==4.17.0
pip install surgeon-pytorch
pip install matplotlib
pip install seaborn
pip install torch
pip install xgboost
pip install colorama #==0.4.6
pip install bayesian-optimization #==1.4.0
pip install typing-extensions
pip install datasets
pip install evaluate
#pip install optuna
#pip install spacy
#pip install git+https://github.com/optuna/optuna.git

#pip install -U pip setuptools wheel
#python -m spacy download en_core_web_sm