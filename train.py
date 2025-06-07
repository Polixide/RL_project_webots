import numpy as np
from gymnasium import Env, spaces
from stable_baselines3 import SAC
from webots_remote_env import WebotsRemoteEnv


env = WebotsRemoteEnv()
model = SAC("MlpPolicy", env, verbose=1 ,device='cuda')
print(f"Using: {model.device}")
model.learn(total_timesteps=500,progress_bar=True)
model.save("models/SAC_minimal_tesla_2.mdl")
env.close()
