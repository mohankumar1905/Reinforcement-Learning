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


class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1

    def partial_fit(self, X, Y):
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

class FeatureTransformer:
    def __init__(self, env):
        #observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        observation_examples = np.random.random((20000, 4)) *2 - 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)
        #converting states into a featurized representations
        #Number of components is number of exemplars
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000))
        ])
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        #print("observations", observations)
        return self.featurizer.transform(self.scaler.transform(observations))

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimensions)
            self.models.append(model)
    
    def predict(self, s):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        return np.array([m.predict(X)[0] for m in self.models]) 

    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])
    
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
        
        if done:
            reward = -200

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

    N = 500
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