import gym
env = gym.make('Taxi-v3').env
print(env.action_space)
print(env.observation_space)

env.reset()
env.s = 128
env.render()

# actions = [1,3,4,0,0,2,2,2,0,0,5]

# for action in actions:
#     observation, reward, done, info = env.step(action)
#     env.render()
    
#     if done: print("Episode finished")

penalties = 0

for i in range(10000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    #env.render()
    # print("observation", observation)
    # print("reward", reward)
    # print("done", done)
    # print("info", info)

    if reward == -10:
        penalties += 1

    if done:
        env.render()
        print("Episode finished after", i, "random actions")
        print("Penalties", penalties)
        break
