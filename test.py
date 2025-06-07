import gymnasium as gym
from webots_remote_env import WebotsRemoteEnv
from stable_baselines3 import SAC

env = WebotsRemoteEnv()
model = SAC.load("models/SAC_minimal_tesla.mdl")
test_episodes = 100

for i in range(test_episodes):
    
    # env.reset() ora restituisce (observation, info)
    observation, _ = env.reset() # Il underscore `_` è una convenzione per ignorare il valore `info`
    
    done = False
    total_reward = 0.0
    while not done:
        action, _states = model.predict(observation=observation, deterministic=True) 
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated # 'done' per Stable-Baselines3 è true se terminated OR truncated
        
        total_reward += reward
        
    print(f"Episode: {i+1} -- Total Reward: {total_reward}")