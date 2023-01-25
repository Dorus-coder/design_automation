from enviroment import ShipEnv
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from gym.wrappers.time_limit import TimeLimit


time_steps = 100
episode_length = 25


env = ShipEnv()
env = DummyVecEnv([lambda: TimeLimit(env=env, max_episode_steps=episode_length)]) # wrapping

model = A2C('MlpPolicy', env, verbose = 2)
try:
    model.learn(total_timesteps=time_steps, log_interval=2 ,progress_bar=True)
except ValueError:
    print('ValueError')
    print(model.get_parameters())
A2C_path = os.path.join('Training', 'Saved Models', 'A2Cv1')

model.save(A2C_path)
