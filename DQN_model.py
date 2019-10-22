import os
import gym
import gym_donkeycar
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

#SET UP ENVIRONMENT
os.environ['DONKEY_SIM_PATH'] = f"./DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(1) # "1" is headless

env = gym.make("donkey-warehouse-v0")
#gym.make("donkey-generated-roads-v0")

timesteps = 100000 # Set this to a reasonable number
model_name = "dqn_model" # Change the model name to your preferences
training = True # Change this to test or use the model

if training:
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(model_name)
else:
    model = DQN.load(model_name)
    obv = env.reset()
    for t in range(10000):
        action, _states = model.predict(obv) # drive straight with small speed
    # execute the action
        obv, reward, done, info = env.step(action)
