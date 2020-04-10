from .agent import Agent

class RandomAgent(Agent):
    
    def get_action(self, observation, reward, done):
        return self.action_space.sample()
