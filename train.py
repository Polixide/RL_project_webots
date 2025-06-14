from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from webots_remote_env import WebotsRemoteEnv
from stable_baselines3 import DDPG

def make_env(port):
    def _init():
        return WebotsRemoteEnv(port=port)
    return _init

if __name__ == '__main__':
    # Supponiamo 4 ambienti sulle porte 10000-10003
    ports = [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007]
    envs = SubprocVecEnv([make_env(p) for p in ports])
    
    #env_test = WebotsRemoteEnv(10000)
    model = PPO("MlpPolicy", envs , device='cuda',batch_size=512,n_steps=256)
    print("PPO loaded")
    
    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("models/PPO_parallel_1M.mdl")
    envs.close()
