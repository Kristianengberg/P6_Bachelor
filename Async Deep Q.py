import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import asyncio

class DQN:
    def __init__(self, input_size, output_size, env):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.learning_decay = 0.01
        self.env = env
        self.model = self._neural_network()

    def _neural_network(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.input_size, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        # model.add(Dense(24, activation='tanh'))
        model.add(Dense(self.output_size, activation='linear'))
        model.compile(loss='MSE', optimizer=Adam(lr=self.learning_rate, decay=self.learning_decay))
        return model

    def train_network(self, state, new_state, reward, action, done):
        y_target = self.model.predict(state)
        y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(new_state)[0])
        self.model.fit(state, y_target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def action(self, state):
        if np.random.random() <= self.epsilon:
            action = env.action_space.sample()
        else:
            action_value = self.model.predict(state)
            action = np.argmax(action_value)
        return action


env = gym.make('CartPole-v0')
network = DQN(env.observation_space.shape[0], env.action_space.n, env)

async def learner_thread(network):
    episodes = 1000
    steps = 500
    mean_reward = 0
    env = gym.make('CartPole-v0')

    for i in range(episodes):

        state = env.reset()
        state = state.reshape(1, 4)
        reward_per_play = 0
        if i % 100 == 0:
            mean_reward = mean_reward / 100
            print("Episode: ", i, "Average Life time for last 100 Episodes: ", mean_reward)
            mean_reward = 0
        for j in range(steps):
            #env.render()
            action = network.action(state)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, 4)

            network.train_network(state, new_state, action, reward, done)

            state = new_state

            mean_reward += reward
            reward_per_play += reward

            if done:
        # print("Total steps for episode ", i, "is ", j)
                break

        await asyncio.sleep(0.0001)

loop = asyncio.get_event_loop()

tasks = [
    asyncio.ensure_future(learner_thread(network)),
    asyncio.ensure_future(learner_thread(network)),
    asyncio.ensure_future(learner_thread(network)),
    asyncio.ensure_future(learner_thread(network)),
]

loop.run_until_complete(asyncio.wait(tasks))
loop.close()