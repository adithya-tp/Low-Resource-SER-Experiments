# %%bash

# pip install -r requirements.txt

env LC_ALL=C.UTF-8
env LANG=C.UTF-8 
env TRANSFORMERS_CACHE=/content/cache
env HF_DATASETS_CACHE=/content/cache
env CUDA_LAUNCH_BLOCKING=1

env WANDB_WATCH=all
env WANDB_LOG_MODEL=1
env WANDB_PROJECT='SER'
wandb login ed9ffb8dd75b1d7d40cd9c9064179d7808a151ed --relogin
