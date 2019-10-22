# Robotics at Berkeley, Learning to drive a car around!

Start small, and build up from there

## Step 1: Downloading:

* please git clone this locally
* also download the corresponding file to your machine, and unzip and place in this directory
  * https://github.com/tawnkramer/gym-donkeycar/releases

## Step 2: Creating a virtual environment

Preq: please install python 3

Q: Why do we use a virtual environment? We use it to install a lot of complex
tools ontop of your system so we don't have to deal with version collisions.

```
# Run this once to set up the VE
>>> python3 -m venv env
# Run this every time you want to use this
>>> source env/bin/activate
```

You will notice that you will have a shell that now looks like:
```
(env) My-Name: directory __
```

#Step 3: install the Donkey Car Gym Package

* `pip install git+https://github.com/tawnkramer/gym-donkeycar.git`

# Step 3: run the "drive straight example"
* If you are on Windows change line 7 in drive_straight.py from
```
os.environ['DONKEY_SIM_PATH'] = f"./donkey_sim.app/Contents/MacOS/donkey_sim"
```
to:
```
os.environ['DONKEY_SIM_PATH'] = f"./donkey_sim.exe"
```
Then:
```
python3 drive_straight.py
```
you should see the simulator open!


Other examples:

https://github.com/tawnkramer/gym-donkeycar/tree/master/examples


Download program for your computer:
https://github.com/tawnkramer/gym-donkeycar/releases

import os
import gym
import gym_donkeycar
import numpy as np

#%% SET UP ENVIRONMENT
os.environ['DONKEY_SIM_PATH'] = f"{PATH_TO_APP}/donkey_sim.app/Contents/MacOS/donkey_sim"
os.environ['DONKEY_SIM_PORT'] = str(9091)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless

env = gym.make("donkey-warehouse-v0")

#%% PLAY
obv = env.reset()
for t in range(100):
    action = np.array([0.0,0.5]) # drive straight with small speed
# execute the action
obv, reward, done, info = env.step(action)
