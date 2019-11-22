import os
import gym
import gym_donkeycar
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import CnnPolicy
from stable_baselines import DDPG

#SET UP ENVIRONMENT
os.environ['DONKEY_SIM_PATH'] = f"../DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless

env = gym.make("donkey-warehouse-v0")
#gym.make("donkey-generated-roads-v0")

timesteps = 100000 # Set this to a reasonable number
model_name = "ddpg_model" # Change the model name to your preferences
training = True # Change this to test or use the model

if training:
    vec_env = DummyVecEnv([lambda: env])
    model = DDPG(CnnPolicy, vec_env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(model_name)
else:
    model = DDPG.load(model_name)
    obv = env.reset()
    for t in range(10000):
        action, _states = model.predict(obv) # drive straight with small speed
    # execute the action
        obv, reward, done, info = env.step(action)
