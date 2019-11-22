import gym
import numpy as np
from setup import load_env
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from pprint import PrettyPrinter
import random
import gym_donkeycar # Registers the environment
import cv2
from matplotlib import pyplot as plt

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from tensorflow.keras.optimizers import Adam

from keras_modal import get_model

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
        # obv: 120x160 matrix
        # reward: 1.001323546455871
        # info: {'pos': (49.99929, 0.7416418, 49.99545), 'cte': -0.000402563, 'speed': 0.02170224, 'hit': 'none'}
        # done: False

        edges = cv2.Canny(obv, 100, 200)
        plt.imshow(edges)
        plt.show()

def rl():
    env = gym.make("donkey-warehouse-v0")
    np.random.seed(123)
    env.seed(123)
    shape = env.action_space.shape
    print(f"Action space shape: {shape}, low:{env.action_space.low}, high:{env.action_space.high}")
    # Action space shape: (2,), low:[-1.  0.], high:[1. 5.]
    nb_actions = shape[0]
    if len(shape) > 1:
        nb_actions *= shape[1]
    print(f"Number of actions: {nb_actions}")

    model = get_model((1,) + env.observation_space.shape, nb_actions)

    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
    dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

    dqn.test(env, nb_episodes=5, visualize=True)
    # Fails because the action returned to the environment is a scalar instead of being a tuple throttle / angle

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

def image_processing():
    from imaging.ethan_burrell.edge import detect_edge

    images = detect_edge("./imaging/ethan_burrell/images/frame_000186_ttl_0_agl_-0.06721038_mil_0.0.jpg")

    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(5, 5))
    for ind, p in enumerate(images):
        ax = axs[ind // 2, ind % 2]
        ax.set_title(p[0])
        ax.imshow(p[1])

    plt.show()

    plt.show()

# drive_straight()
# cnn()
# image_processing()
rl()
