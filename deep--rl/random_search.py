#export PATH=~/anaconda3/bin:$PATHH
import gym
from gym import wrappers
import numpy as np
from matplotlib import pyplot as plt

def get_action(s, w):
    '''Returns action 1 if the current state multiplied by the weights give you greater than zero'''
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0
    while not done and t < 10000:
        t += 1
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        if done:
            break
    return t #how many iterations the game has been running based on the action we took

def play_multiple_episodes(env, T, params):
    '''Plays the epsiode T number of times and stores episode length of each iterations'''
    episode_lengths = np.empty(T)
    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)
    avg_length = episode_lengths.mean()
    print("Average Length: ", avg_length)
    return avg_length

def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4) * 2 - 1
        avg_length = play_multiple_episodes(env, 100, new_params)
        episode_lengths.append(avg_length)

        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params

def main():
    env = gym.make('CartPole-v0')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()

    print("**Play with best weights for 100 times")
    env  = wrappers.Monitor(env, 'video', force=True)
    play_one_episode(env, params)

if __name__ == "__main__":
    main()