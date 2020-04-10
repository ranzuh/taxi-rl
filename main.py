import gym
from agents.random_agent import RandomAgent

env = gym.make('Taxi-v3')

observation = env.reset()
reward = 0
done = False
penalties = 0
episodes = 10000

agent = RandomAgent(env.action_space)
env.render()

for i in range(episodes):
    
    action = agent.get_action(observation, reward, done)
    observation, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    if done:
        env.render()
        print("Episode finished after", i, "random actions")
        print("Penalties", penalties)
        break
