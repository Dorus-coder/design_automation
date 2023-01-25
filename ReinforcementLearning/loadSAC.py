#%%
from enviroment import ShipEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from gym.wrappers.time_limit import TimeLimit

time_steps = 100
episode_length = 25

env = ShipEnv()
env = DummyVecEnv([lambda: TimeLimit(env=env, max_episode_steps=episode_length)]) # wrapping

#%% 
cwd = os.getcwd()
abs_path = cwd + r'\ReinforcementLearning\Training\Saved Models\SACv0'

model = SAC.load(abs_path, env=env, verbose = 1, normalize_advantage=True) # tensorboard_log=log_path


# %%
# model.learn(total_timesteps=time_steps, log_interval=2 ,progress_bar=True)
# from stable_baselines3.common.evaluation import evaluate_policy
# evalpl = evaluate_policy(model, env, n_eval_episodes=10, render=True)
# evalpl (0, 0) after training for 100 steps    
# %%

SAC_path = os.path.join('Training', 'Saved Models', 'SACv0')

model.save(SAC_path)

episodes = 10
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0 
    len_iterations = 0
    print("."*50)
    while not done:
        len_iterations += 1
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score+=reward
        print(info)
    print(f'Episode:{episode} Score:{score} Iterations:{len_iterations}')
    print("_"*50)