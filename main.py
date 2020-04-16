import gym
from agents.random_agent import RandomAgent
from agents.qlearning_agent import QLearningAgent

env = gym.make('Taxi-v3')

total_penalties = 0
episodes = 2001
completions = 0

agent = RandomAgent(env.action_space)
agent2 = QLearningAgent(env.action_space, env.observation_space)

for episode in range(episodes):
    if episode % 1000 == 0:
        print(episode)
    observation = env.reset()
    reward = 0
    done = False
    penalties = 0
    # env.render()
    for i in range(200):
        # env.render()
        action = agent2.get_action(observation, reward, done)
        observation, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        if done:
            agent2.get_action(observation, reward, done)
            # env.render()
            # print("Episode finished after", i, "time steps")
            # print("Penalties occured", penalties)
            #
            if reward == 20:
                # print("Objective completed")
                completions += 1
            # else:
            #     print("Objective not completed")

            break
    total_penalties += penalties

agent2.save_table()

print()
print("Average penalties over episode", total_penalties / episodes)
print("Completions", completions)

total_penalties = 0
episodes = 200
completions = 0

for episode in range(episodes):
    observation = env.reset()
    reward = 0
    done = False
    penalties = 0
    # env.render()
    for i in range(200):
        # env.render()
        action = agent2.get_learned_action(observation)
        observation, reward, done, info = env.step(action)


        if reward == -10:
            penalties += 1

        if done:
            # print(reward)
            # print(reward)
            # env.render()
            # print("Episode finished after", i, "time steps")
            # print("Penalties occured", penalties)
            #
            if reward == 20:
                # print("Objective completed")
                completions += 1
            # else:
            #     print("Objective not completed")

            break
    total_penalties += penalties

print()
print("Average penalties over episode", total_penalties / episodes)
print("Completions", completions)

#agent2.printq()
