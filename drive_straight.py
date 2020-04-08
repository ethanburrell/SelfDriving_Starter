import os
import gym
import gym_donkeycar
import numpy as np

#%% SET UP ENVIRONMENT
os.environ['DONKEY_SIM_PATH'] = f"./donkey_sim.app/Contents/MacOS/donkey_sim"
#"donkey_sim.app"
#f"{PATH_TO_APP}/donkey_sim.app/Contents/MacOS/donkey_sim"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless

env = gym.make("donkey-warehouse-v0")
#gym.make("donkey-generated-roads-v0")

#%% PLAY
obv = env.reset()
for t in range(10000):
    action = np.array([0.0,0.5]) # drive straight with small speed
# execute the action
    obv, reward, done, info = env.step(action)
