import gym
from agents.random_agent import RandomAgent
from agents.qlearning_agent import QLearningAgent

env = gym.make('Taxi-v3')

total_penalties = 0
episodes = 10001
completions = 0

agent = RandomAgent(env.action_space)
agent2 = QLearningAgent(env.action_space, env.observation_space)

for episode in range(episodes):
    if episode % 100 == 0:
        print(episode)
    state = env.reset()
    reward = 0
    done = False
    penalties = 0
    # env.render()
    while not done:
        # env.render()
        action = agent2.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent2.update(state, action, next_state, reward)

        state = next_state

        if reward == -10:
            penalties += 1

    total_penalties += penalties

agent2.save_table()

print()
print("Average penalties over episode", total_penalties / episodes)
print("Completions", completions)

total_penalties = 0
episodes = 1
completions = 0

for episode in range(episodes):
    state = env.reset()
    reward = 0
    done = False
    penalties = 0
    env.render()
    while not done:
        env.render()
        import sys
        sys.stdout.flush()
        action = agent2.get_policy(state)
        next_state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        state = next_state
    total_penalties += penalties

print()
print("Average penalties over episode", total_penalties / episodes)
print("Completions", completions)

# agent2.printq()
