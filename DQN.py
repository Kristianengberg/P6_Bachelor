import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

env = gym.make("CartPole-v0")

e = 1.0
e_decay = 0.995
gamma = 0.95
episode = 1000
steps = 500
learning_rate = 0.001
memory = deque(maxlen=2000)
batch_size = 32
input_shape = env.observation_space.shape[0]

done = False


def q_network():
    model = Sequential()
    model.add(Dense(24, input_dim=input_shape, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


agent = q_network()


def experience_replay(e):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma *
                      np.amax(agent.predict(next_state)[0]))
        target_f = agent.predict(state)
        target_f[0][action] = target
        agent.fit(state, target_f, epochs=1, verbose=0)


def add_memory(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def act(state):
    if np.random.rand() <= e:
        return random.randrange(env.action_space.n)
    act_values = agent.predict(state)
    return np.argmax(act_values[0])  # returns action

for i in range(episode):
    obs = env.reset()
    obs = obs.reshape(1, 4)
    for j in range(steps):
        # env.render()
        action = act(obs)
        new_obs, reward, done, _ = env.step(action)

        new_obs = obs.reshape(1, 4)

        reward = reward if not done else -10

        add_memory(obs, action, reward, new_obs, done)

        obs = new_obs

        if done:
            if e >= 0.01:
                e *= e_decay

            print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(i, episode, j, e))
            break
    if len(memory) > batch_size:
        experience_replay(e)

    # print(all_rewards)
