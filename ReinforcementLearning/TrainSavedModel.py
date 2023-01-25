from stable_baselines3 import SAC
from enviroment import ShipEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import os

time_steps = 1000

env = ShipEnv()
env = DummyVecEnv([lambda: env]) # wrapping
cwd = os.getcwd()
abs_path = cwd + r'\ReinforcementLearning\Training\Saved Models\SACv0'
model = SAC.load(abs_path, env=env, verbose=1) # tensorboard_log=log_path

# from stable_baselines3.common.evaluation import evaluate_policy
# perf = evaluate_policy(model, env, n_eval_episodes=5, render=False)
# print(f"{perf = }")

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


