from enviroment import ShipEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from gym.wrappers.time_limit import TimeLimit


time_steps = 100
episode_length = 25


env = ShipEnv()
env = DummyVecEnv([lambda: TimeLimit(env=env, max_episode_steps=episode_length)]) # wrapping

model = PPO('MlpPolicy', env, verbose = 1, normalize_advantage=True)
model.learn(total_timesteps=time_steps, log_interval=2 ,progress_bar=True)

PPO_path = os.path.join('Training', 'Saved Models', 'PPOv0')

model.save(PPO_path)
