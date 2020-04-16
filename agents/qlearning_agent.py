from .agent import Agent
import numpy as np

np.set_printoptions(threshold=np.inf)


class QLearningAgent(Agent):
    # learning rate
    alpha = 0.1

    # how often random move
    epsilon = 0.1

    # discount future rewards
    discount = 0.6

    def __init__(self, action_space, observation_space):
        super().__init__(action_space)

        # initialize empty Q-table
        self.Q = np.zeros((observation_space.n, action_space.n))

        # initialize state, action and reward
        self.s = self.a = self.r = None

    def get_action(self, next_state, reward, done):
        # print(observation,reward,done)

        Q, s, a, r = self.Q, self.s, self.a, self.r
        alpha, epsilon, discount = self.alpha, self.epsilon, self.discount

        if done:
            Q[s, None] = reward

        if s is not None:
            Q[s, a] = Q[s, a] + alpha * (r + discount * (np.max(Q[next_state])) - Q[s, a])
            # print(self.q[observation][self.a])

        if done:
            self.s = self.a = self.r = None
        else:
            self.s, self.r = next_state, reward

            # Epsilon-greedy action selection
            if np.random.random_sample() < epsilon:
                self.a = self.action_space.sample()
            else:
                self.a = np.argmax(Q[next_state])

        return self.a

    def get_learned_action(self, observation):
        # print(self.q[observation])
        # print(np.argmax(self.q[observation]))
        return np.argmax(self.Q[observation])

    def printq(self):
        print(self.Q)

    def save_table(self):
        np.savetxt("table.csv", self.Q, delimiter=",")

    def load_table(self):
        self.Q = np.loadtxt(open("table.csv"), delimiter=",")
