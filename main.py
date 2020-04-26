import gym
from agents.random_agent import RandomAgent
from agents.qlearning_agent import QLearningAgent
import numpy as np


def train(env, agent, episodes=10001):
    total_penalties = 0
    completions = 0

    timesteps_per_episode = []
    rewards_per_episode = []

    for episode in range(episodes):
        if episode % 100 == 0:
            print("Episode:", episode)

        state = env.reset()
        reward = 0
        done = False
        penalties = 0
        total_reward = 0
        timesteps = 0
        # env.render()
        while not done:
            # env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, next_state, reward)

            state = next_state
            total_reward += reward

            if reward == -10:
                penalties += 1

            if reward == 20:
                completions += 1

            timesteps += 1

        total_penalties += penalties

        if episode % 100 == 0:
            rewards_per_episode.append(total_reward)
            timesteps_per_episode.append(timesteps)

    print()
    print("Training complete after", episodes, "episodes")
    print("Average penalties over episode", total_penalties / episodes)
    print("Completions", completions)
    print()


def evaluate(env, agent, episodes=100):
    total_penalties = 0
    completions = 0

    # these are for statistics
    rewards_per_episode = []
    timesteps_per_episode = []
    penalties_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        reward = 0
        done = False
        penalties = 0
        total_reward = 0
        timesteps = 0
        # env.render()
        while not done:
            action = agent.get_policy(state)
            next_state, reward, done, info = env.step(action)

            state = next_state

            total_reward += reward
            timesteps += 1

            if reward == -10:
                penalties += 1

            if reward == 20:
                completions += 1

            # render_frame()

        penalties_per_episode.append(penalties)
        rewards_per_episode.append(total_reward)
        timesteps_per_episode.append(timesteps)

    # print()
    # print("Evaluated agent for", episodes, "episodes")
    # print("Average penalties over episode", total_penalties / episodes)
    # print("Completions", completions)
    # print()

    return np.mean(rewards_per_episode), np.mean(penalties_per_episode), np.mean(timesteps_per_episode)


if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    agent = QLearningAgent(env.action_space, env.observation_space)
    train(env, agent, 10001)
    print("Q-learning agent")
    evaluate(env, agent, 100)

    print("Random agent")
    rand_agent = RandomAgent(env.action_space)
    evaluate(env, rand_agent, 100)
