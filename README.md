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
$ python3 -m venv env
```
Every time you want to start the virtual environment
```
on mac
$ source env/bin/activate
on windows
$ source env/Scripts/activate
```

You will notice that you will have a shell that now looks like:
```
(env) My-Name: directory __
```

#Step 3: install the Donkey Car Gym Package

* `pip install git+https://github.com/tawnkramer/gym-donkeycar.git`

# Step 4: run the "drive straight example"

* If you are on Windows change line 7 in drive_straight.py from
```
os.environ['DONKEY_SIM_PATH'] = f"./donkey_sim.app/Contents/MacOS/donkey_sim"
```
to:
```
os.environ['DONKEY_SIM_PATH'] = f"./DonkeySimWindows/DonkeySim.exe"

(make sure this is correct path to where the exe lives)
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
