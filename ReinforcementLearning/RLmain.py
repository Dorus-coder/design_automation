from enviroment import ShipEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from gym.wrappers.time_limit import TimeLimit
time_steps = 100
episode_length = 25
# TimeLimit(env, episode_length)]
env = ShipEnv()
env = DummyVecEnv([lambda: TimeLimit(env=env, max_episode_steps=episode_length)]) # wrapping

# from stable_baselines3.common.env_checker import check_env


# It will check your custom environment and output additional warnings if needed
# check_env(env)

# policies = ['MultiInputPolicy']
model = PPO.load('Training\Saved Models\PPOv0_model.zip', env=env, verbose = 1, normalize_advantage=True) # tensorboard_log=log_path

# model = A2C('MlpPolicy', env, verbose = 1, normalize_advantage=True) # tensorboard_log=log_path

# # model = A2C('MlpPolicy', env, verbose = 1) # tensorboard_log=log_path


model.learn(total_timesteps=time_steps, log_interval=2 ,progress_bar=True)


PPO_path = os.path.join('Training', 'Saved Models', 'A2C_model')

model.save(PPO_path)