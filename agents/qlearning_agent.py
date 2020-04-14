from .agent import Agent
import numpy as np
np.set_printoptions(threshold=np.inf)

class QLearningAgent(Agent):
    # stepsize
    alpha = 0.1

    epsilon = 0.3

    discount = 0.9

    def __init__(self, action_space, observation_space):
        super().__init__(action_space)
        self.q = np.zeros((observation_space.n, action_space.n))
        self.s = None
        self.a = None
        self.r = None

    def get_action(self, observation, reward, done):
        #print(observation,reward,done)

        if self.s is not None:
            self.q[self.s][self.a] = self.q[self.s][self.a] + self.alpha*(self.r + self.discount * (np.max(
                self.q[observation])) - self.q[self.s][self.a])
            #print(self.q[observation][self.a])

        self.s, self.r = observation, reward

        # Epsilon-greedy action selection
        if np.random.random_sample() <= self.epsilon:
            self.a = self.action_space.sample()
        else:
            self.a = np.argmax(self.q[observation])

        return self.a

    def printq(self):
        print(self.q)

