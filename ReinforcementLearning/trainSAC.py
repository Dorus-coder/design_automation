"""
Note:

    The default policies for SAC differ a bit from others MlpPolicy: 
    it uses ReLU instead of tanh activation, to match the original paper.


MlpPolicy 	Policy object that implements actor critic, using a MLP (2 layers of 64)
MlpLstmPolicy 	Policy object that implements actor critic, using LSTMs with a MLP feature extraction
MlpLnLstmPolicy 	Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction
CnnPolicy 	Policy object that implements actor critic, using a CNN (the nature CNN)
CnnLstmPolicy 	Policy object that implements actor critic, using LSTMs with a CNN feature extraction
CnnLnLstmPolicy 	Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction

"""


from enviroment import ShipEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from gym.wrappers.time_limit import TimeLimit


time_steps = 100
episode_length = 25


env = ShipEnv()
env = DummyVecEnv([lambda: TimeLimit(env=env, max_episode_steps=episode_length)]) # wrapping

model = SAC('MlpPolicy', env, verbose = 2)
try:
    model.learn(total_timesteps=time_steps, log_interval=2 ,progress_bar=True)
except ValueError:
    print('ValueError')
    print(model.get_parameters())
SAC_path = os.path.join('Training', 'Saved Models', 'SACv0')

model.save(SAC_path)
