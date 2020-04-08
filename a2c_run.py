import os
import gym
import gym_donkeycar
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from stable_baselines.results_plotter import *
from stable_baselines.bench import Monitor
from stable_baselines import DDPG, A2C, PPO2

best_reward = -np.inf
num_steps = 0

def callback(_locals, _globals):
    global best_reward, num_steps
    if (num_steps + 1) % 500 == 0:
        x, y = ts2xy(load_results("models/"), 'timesteps')
        if len(x) > 0:
            curr_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            if curr_reward > best_reward:
                best_reward = curr_reward
                print("Saving model")
                _locals['self'].save("models/a2c.pkl")
    num_steps += 1
    return True

def make(env_str, cpu):
    def make_helper():
        env = gym.make(env_str)
        env.seed(cpu)
        env = Monitor(env, "models/", allow_early_resets=True)
        env.reset()
        return env
    return make_helper


#SET UP ENVIRONMENT
os.environ['DONKEY_SIM_PATH'] = f"./donkey_sim.app/Contents/MacOS/donkey_sim"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless
os.environ['DONKEY_SIM_MULTI'] = str(1)
os.makedirs("models/", exist_ok=True)

timesteps = 100000 # Set this to a reasonable number
model_name = "a2c_model" # Change the model name to your preferences
training = True # Change this to test or use the model
num_cpu = 1

if training:
    #env = DummyVecEnv([lambda: env])
    #
    # donkey-warehouse-v0
    env = SubprocVecEnv([make("donkey-generated-roads-v0", i) for i in range(num_cpu)])
    model = A2C(CnnLstmPolicy, env, verbose=1)
    print("Training")
    model.learn(timesteps, callback=callback)
    obs=env.reset()
    print("Testing")
    for i in range(5000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        try:
            env.render()
        except Exception as e:
            print(e)
        if i % 1 == 0:
            print("Saving model")
            #model.save(model_name)
            _locals['self'].save('best_model.pkl')
    #model.save(model_name)
    _locals['self'].save('best_model.pkl')
else:
    env = SubprocVecEnv([make("donkey-generated-roads-v0", i) for i in range(num_cpu)])
    model = A2C.load("a2c_model.pkl")
    obs=env.reset()
    print("Testing")
    for i in range(5000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        try:
            env.render()
        except Exception as e:
            print(e)
