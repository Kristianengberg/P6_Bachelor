import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class DQN:
    def __init__(self, input_size, output_size, env):
        self.input_size = input_size
        self.output_size = output_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.learning_decay = 0.01
        self.env = env
        self.model = self._neural_network()

    def replay_memory(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def _neural_network(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.input_size, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        # model.add(Dense(24, activation='tanh'))
        model.add(Dense(self.output_size, activation='linear'))
        model.compile(loss='MSE', optimizer=Adam(lr=self.learning_rate, decay=self.learning_decay))
        return model

    def train_network(self, batch_size):
        x_batch, y_batch = [], []
        if len(self.memory) >= batch_size:
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, new_state, done in minibatch:
                y_target = self.model.predict(state)
                y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(new_state)[0])
                x_batch.append(state[0])
                y_batch.append(y_target[0])

            self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), epochs=1, verbose=0)
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
agents = []
amount_agents = 5
for _ in range(amount_agents):
    agents.append(DQN(env.observation_space.shape[0], env.action_space.n, env))

episodes = 1000
steps = 500
batch_size = 64
mean_reward = 0
plot_reward = []

for agent in range(len(agents)):
    for i in range(episodes):
        state = env.reset()
        state = state.reshape(1, 4)
        reward_per_play = 0
        if i % 100 == 0:
            mean_reward = mean_reward / 100
            print("Episode: ", i, "Average Life time for last 100 Episodes: ", mean_reward)
            mean_reward = 0
        for j in range(steps):
            # env.render()
            action = agents[agent].action(state)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, 4)
            agents[agent].replay_memory(state, action, reward, new_state, done)
            state = new_state

            mean_reward += reward
            reward_per_play += reward

            if done:
        # print("Total steps for episode ", i, "is ", j)
                break
        agents[agent].train_network(batch_size)

        if agent == 0:
            plot_reward.append(float(reward_per_play))
        else:
            plot_reward[i] += reward_per_play
print(plot_reward[199])
plot_reward_two = []
for rewards in range(episodes):
    plot_reward_two.append(float(plot_reward[rewards] / amount_agents))
plt.plot(plot_reward_two)
print(plot_reward_two[199])
plt.show()

# print("Current Epsilon", agent.epsilon)
