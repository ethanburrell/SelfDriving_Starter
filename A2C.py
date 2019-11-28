import os
import gym
import gym_donkeycar
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from stable_baselines import DDPG, A2C

#SET UP ENVIRONMENT
os.environ['DONKEY_SIM_PATH'] = f"../DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless

env = gym.make("donkey-warehouse-v0")

timesteps = 100000 # Set this to a reasonable number
model_name = "a2c_model" # Change the model name to your preferences
training = True # Change this to test or use the model

if training:
    env = DummyVecEnv([lambda: env])
    model = A2C(CnnPolicy, env, verbose=1)
    model.learn(timesteps)
    obs=env.reset()
    for i in range(5000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        try:
            env.render()
        except Exception as e:
            print(e)
        if i % 50 == 0:
            model.save(model_name)
    model.save(model_name)