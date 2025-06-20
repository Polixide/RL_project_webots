import numpy as np
from gymnasium import Env, spaces
from stable_baselines3 import SAC
from webots_remote_env import WebotsRemoteEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import os

wandb.init(
    project="RL_project_webots_easy",          
    name="SAC-Webots-run",             
    sync_tensorboard=True,             
    monitor_gym=True,                  
    save_code=True
)
# Percorsi per salvataggio
CHECKPOINT_DIR = "checkpoints"
MODEL_DIR = "models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Callback per salvataggi periodici
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,                     # salva ogni 100k timesteps
    save_path=CHECKPOINT_DIR,
    name_prefix="SAC_run1_"
)

# Callback per integrazione con wandb
wandb_callback = WandbCallback(
    gradient_save_freq=1,
    model_save_path=MODEL_DIR,
    verbose=2,
)

env = WebotsRemoteEnv()
model = SAC("MlpPolicy", env, verbose=1 ,device='cuda',tensorboard_log="./tb_logs/")
print(f"Using: {model.device}")
model.learn(total_timesteps=1000000,progress_bar=True,callback=[checkpoint_callback, wandb_callback])
model.save(os.path.join(MODEL_DIR, "SAC_1M_1.mdl"))
env.close()
wandb.finish()