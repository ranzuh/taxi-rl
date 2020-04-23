from .agent import Agent


class RandomAgent(Agent):

    def get_action(self, state):
        return self.action_space.sample()

    def get_policy(self, state):
        return self.get_action(state)
