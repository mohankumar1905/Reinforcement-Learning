import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from qlearning_with_bins import plot_running_avg
import warnings
#from tfmodel import ANN1
from ptmodel import ANN1
warnings.filterwarnings("ignore")


class FeatureTransformer:
    def __init__(self, env):
        #observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        observation_examples = np.random.random((20000, 4)) *2 - 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)
        #Number of components is number of exemplars
        self.scaler = scaler
    def transform(self, observations):
        #print("observations", observationANN1s)
        return self.scaler.transform(observations)

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = ANN1()
            self.models.append(model)
    
    def predict(self, s):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        return np.array([m.predict(X)[0] for m in self.models]) 

    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, np.array(G))
    
    def sample_action(self, s, eps):
        #using optimisitic initial values - this time.
        if np.random.random() < eps: #should be eps for epsilon greedy
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(model, eps, gamma, env):
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0
    while not done and iters < 2000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        
        if done and iters < 199:
            reward = -500

        #update the model
        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G)

        if reward == 1:
            total_reward += reward
        iters+=1
    return total_reward

def main(monitor = False):
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99

    if monitor:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 1/np.sqrt(n+1)
        total_reward = play_one(model, eps, gamma, env)
        total_rewards[n] = total_reward
        print("episode: ", n, "total reward", total_reward)
    print("average reward for last 100 episodes: ", total_rewards[-100:].mean())
    print("total_steps: ", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)

if __name__ == "__main__":
    if "monitor" in sys.argv:
        main(monitor = True)
    else:
        main()