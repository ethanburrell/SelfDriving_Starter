import gym
import numpy as np
from setup import load_env
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from pprint import PrettyPrinter
import random

load_env()

pp = PrettyPrinter()

def drive_straight():
    env = gym.make("donkey-warehouse-v0")
    # gym.make("donkey-generated-roads-v0")

    #%% PLAY
    obv = env.reset()
    for t in range(10000):
        action = np.array([0.0,0.5]) # drive straight with small speed
    # execute the action
        obv, reward, done, info = env.step(action)

def drive_with_prediction():
    env = gym.make("donkey-warehouse-v0")
    obv = env.reset()
    for t in range(1000):
        action = get_action(env)
        obv, reward, done, info = env.step(action)


# gets an action given the environment.

def get_action(env):
    epsilon = 0.2 #probability that we will pick a random action rather than the best one.
    if random.uniform(0, 1) < epsilon: 
        return env.action_space.sample()
    else:
        #Get the actual action!
        return env.action_space.high
    

def cnn():
    # totally random wtf init
    # this is a template to use on the Q learning method
    x_train = x_test = np.random.random((1000, 28, 28, 1)) # much overfitting xD
    y_train = y_test = np.random.randint(2, size=(1000, 1))

    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1, activation="softmax", input_shape=(18432,)))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

    score = model.evaluate(x_test, y_test, batch_size=128)
    print(f"Score: {score}")

    #predict first 4 data points in the test set
    print("Prediction:")
    pp.pprint(model.predict(x_test[:4]))

    #actual results for first data points in test set
    print("vs reality:")
    pp.pprint(y_test[:4])

#drive_straight()
cnn()
