import numpy as np
from gymnasium import Env, spaces
from stable_baselines3 import SAC
from webots_remote_env import WebotsRemoteEnv


env = WebotsRemoteEnv()
model = SAC("MlpPolicy", env, verbose=1 ,device='cuda')
print(f"Using: {model.device}")
model.learn(total_timesteps=50000,progress_bar=True)
model.save("models/SAC_minimal_tesla.mdl")
env.close()
