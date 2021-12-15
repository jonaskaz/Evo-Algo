"""
This implementation was adapted from:
https://medium.com/swlh/introduction-to-q-learning-with-openai-gym-2d794da10f3d
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time


class Taxi:
    def __init__(self, hyperparams: dict):
        self.env = gym.make('Taxi-v3')
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.alpha = hyperparams["alpha"]
        self.epsilon = hyperparams["epsilon"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_max = hyperparams["epsilon_max"]
        self.decay = hyperparams["decay"]
        self.discount = hyperparams["discount"]
        
    def run(self, num_episodes):
        self.all_rewards = []
        self.epsilons = []
        for episode in range(num_episodes):
            state = self.env.reset()
            self.run_episode(state)
            self.update_epsilon(episode)
        self.plot(num_episodes)

    def run_episode(self, state, simulate=False):
        rewards=0
        done = False
        while not done:
            exploit = random.uniform(0,1)
            if exploit > self.epsilon:
                action = np.argmax(self.q[state,:])
            else:
                action = self.env.action_space.sample()
            
            new_state, reward, done, _ = self.env.step(action)
            if simulate:
                self.env.render()
                time.sleep(0.8)
            rewards += reward
            self.bellman(state, new_state, action, reward)
            state=new_state
        self.all_rewards.append(rewards)

    def bellman(self, state, new_state, action, reward):
        future_q = self.alpha*(reward + self.discount * np.max(self.q[new_state,:]) - self.q[state, action])
        self.q[state, action] = self.q[state, action] + future_q
    
    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min)*np.exp(-self.decay*episode)
        self.epsilons.append(self.epsilon)

    def plot(self, num_episodes):
        _, axis = plt.subplots(1, 2)
        x = range(num_episodes)
        self.plot_epsilons(axis, x)
        self.plot_rewards(axis, x)
        plt.show()

    def plot_rewards(self, axis, x):
        axis[1].plot(x, self.all_rewards)
        axis[1].set_xlabel('Episode')
        axis[1].set_ylabel('Training total reward')
        axis[1].set_title('Total rewards over all episodes in training') 
    
    def plot_epsilons(self, axis, x):
        axis[0].plot(x, self.epsilons)
        axis[0].set_xlabel('Episode')
        axis[0].set_ylabel('Epsilons')
        axis[0].set_title('Epsilons over all episodes in Training')

    def simulate(self):
        self.epsilon=self.epsilon_min
        state = self.env.reset()
        self.run_episode(state, True)

if __name__ == "__main__":
    hyperparams = {
        "alpha": 0.6,
        "epsilon": 1,
        "epsilon_min": 0.1,
        "epsilon_max": 1,
        "decay": 0.001,
        "discount": 0.7
    }
    taxi = Taxi(hyperparams)
    taxi.run(10000)
    taxi.simulate()