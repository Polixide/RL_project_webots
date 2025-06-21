from stable_baselines3 import SAC
from webots_remote_env import WebotsRemoteEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time


# Inizializza ambiente visualizzabile
env = DummyVecEnv([lambda: WebotsRemoteEnv()])

# Carica il modello
model = SAC.load("models/SAC_1M_UDR.mdl", env=env)

# Parametri test
test_episodes = 200
all_rewards = []

for ep in range(test_episodes):
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(f"ACTION: {action}")
        obs, reward, terminated, _ = env.step(action)
        done = terminated
        total_reward += float(reward)
        steps += 1
    

    all_rewards.append(total_reward)
    print(f"Episode {ep+1} | Reward: {total_reward:.2f} | Steps: {steps}")

print("\nRISULTATI TEST")
print(f"Avg Reward: {np.mean(all_rewards):.2f}")
