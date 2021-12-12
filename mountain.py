import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import json
import ast


class MountainClimber:
    def __init__(self, precision, num_episodes, hyperparams: dict):
        self.env = gym.make("MountainCar-v0")
        self.precision = precision
        self.q = self.create_q()
        self.alpha = hyperparams["alpha"]
        self.epsilon = hyperparams["epsilon"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_max = hyperparams["epsilon_max"]
        self.decay = (self.epsilon - self.epsilon_min) / num_episodes
        self.discount = hyperparams["discount"]
        self.num_episodes = num_episodes

    def create_q(self):
        possible_positions = np.arange(-1.2, 0.61, 10 ** (-1 * self.precision))
        possible_velocities = np.arange(-0.07, 0.07, 10 ** (-1 * self.precision))
        Q = {(0.0, 0.0): [0, 0, 0]}
        for e1 in possible_positions:
            for e2 in possible_velocities:
                Q[(round(e1, self.precision), round(e2, self.precision))] = [
                    0 for i in range(3)
                ]
        return Q

    def run(self):
        self.all_rewards = []
        self.epsilons = []
        for episode in range(self.num_episodes):
            state = self.make_discrete_state(self.env.reset())
            rewards = self.run_episode(state)
            self.all_rewards.append(rewards)
            self.update_epsilon()
            if episode % 100 == 0:
                print(f"Episode {episode} total reward = {rewards}")
        self.plot()

    def run_episode(self, state, simulate=False):
        rewards = 0
        done = False
        while not done:
            exploit = random.uniform(0, 1)
            if exploit > self.epsilon:
                action = np.argmax(self.q[state])
            else:
                action = self.env.action_space.sample()

            new_state, reward, done, _ = self.env.step(action)
            new_state = self.make_discrete_state(new_state)
            if simulate:
                self.env.render()
                time.sleep(0.1)
            rewards += reward
            self.bellman(state, new_state, action, reward)
            state = new_state
        return rewards

    def array2tuple(self, array):
        try:
            return tuple(self.array2tuple(i) for i in array)
        except TypeError:
            return array

    def array_round(self, state):
        pos = round(float(state[0]), self.precision)
        vel = round(float(state[1]), self.precision)
        return pos, vel

    def make_discrete_state(self, state):
        return self.array2tuple(self.array_round(state))

    def bellman(self, state, new_state, action, reward):
        future_q = self.alpha * (
            reward + self.discount * max(self.q[new_state]) - self.q[state][action]
        )
        self.q[state][action] = self.q[state][action] + future_q

    def update_epsilon(self):
        self.epsilon -= self.decay
        self.epsilons.append(self.epsilon)

    def simulate(self):
        self.epsilon = -1
        state = self.env.reset()
        state = self.make_discrete_state(state)
        self.run_episode(state, True)

    def plot(self):
        _, axis = plt.subplots(1, 2)
        x = range(self.num_episodes)
        self.plot_epsilons(axis, x)
        self.plot_rewards(axis, x)
        plt.show()

    def plot_rewards(self, axis, x):
        axis[1].plot(x, self.all_rewards)
        axis[1].set_xlabel("Episode")
        axis[1].set_ylabel("Training total reward")
        axis[1].set_title("Total rewards over all episodes in training")

    def plot_epsilons(self, axis, x):
        axis[0].plot(x, self.epsilons)
        axis[0].set_xlabel("Episode")
        axis[0].set_ylabel("Epsilons")
        axis[0].set_title("Epsilons over all episodes in Training")
    
    def save_q(self, filename):
        with open(filename,'w+') as outfile:
            json.dump({str(k): v for k, v in self.q.items()}, outfile)
        
    def load_q(self, filename):
        with open(filename, 'r') as infile:
            data = json.load(infile)
            self.q = {ast.literal_eval(k): v for k, v in data.items()}


if __name__ == "__main__":
    hyperparams = {
        "alpha": 0.2,
        "epsilon": 1,
        "epsilon_min": 0.01,
        "epsilon_max": 1,
        "discount": 0.9,
    }
    mc = MountainClimber(2, 100000, hyperparams)
    #mc.run()
    #mc.save_q("Q_100k_epis.json")
    mc.load_q("Q_100k_epis.json")
    mc.simulate()
