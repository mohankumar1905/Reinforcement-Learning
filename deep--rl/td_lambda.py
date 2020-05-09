import gym
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from gym import wrappers
from datetime import datetime

from qlearning_with_mountain_car import plot_cost_to_go, FeatureTransformer, plot_running_avg

class BaseModel:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, input_, target, eligibility, lr=10e-3):
        self.w += lr*(target - input_.dot(self.w))*eligibility

    def predict(self, X):
        X = np.array(X)
        return X.dot(self.w)


#Holds one base model for each action
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        D = feature_transformer.dimensions
        self.eligibilities = np.zeros((env.action_space.n, D))
        for i in range(env.action_space.n):
            model = BaseModel(D)
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        return np.array([model.predict(X)[0] for model in self.models])

    def update(self, s, a, G, gamma, lambda_):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.eligibilities *= gamma * lambda_
        self.eligibilities += X[0]
        self.models[a].partial_fit(X[0], G, self.eligibilities[a])

    def sample_action(self, s, eps):
        #using optimisitic initial values - this time.
        if np.random.random() < eps: #should be eps for epsilon greedy
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(model, eps, gamma, lambda_, env):
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0
    while not done and iters < 200:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        #update the model
        G = reward + gamma * np.max(model.predict(observation)[0])
        model.update(prev_observation, action, G, gamma, lambda_)

        total_reward += reward
        iters+=1
    return total_reward


def main(monitor = False):
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99
    lambda_= 0.7

    if monitor:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        total_reward = play_one(model, eps, gamma, lambda_, env)
        total_rewards[n] = total_reward
        print("episode: ", n, "total reward", total_reward)
    print("average reward for last 100 episodes: ", total_rewards[-100:].mean())
    print("total_steps: ", -total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)
    plot_cost_to_go(env, model)

if __name__ == "__main__":
    if "monitor" in sys.argv:
        main(monitor = True)
    else:
        main()