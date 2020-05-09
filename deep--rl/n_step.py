import gym
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from gym import wrappers
from datetime import datetime

import qlearning_with_mountain_car

class SGDRegressor:
    def __init__(self, **kwargs):
        self.w = None
        self.lr = 10e-3

    def partial_fit(self, X, Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)
        
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

#Replace SKlearn regressor
qlearning_with_mountain_car.SGDRegressor = SGDRegressor

def play_one(model, eps, gamma, n=5):
    observation = env.reset()
    done = False
    total_reward = 0
    rewards = []
    states = []
    actions = []
    iters = 0

    #array of [gamma^0, gamma^1, gamma^2, ...... gamma^(n-1)]
    multiplier = np.array([gamma]*n)**np.arange(n)

    while not done and iters<200:
        action = model.sample_action(observation, eps)
        states.append(observation)
        actions.append(action)
        prev_observation = observation
        observation, reward, done,  info = env.step(action)
        rewards = rewards.append(reward)

        #update the model after collecting n rewards 
        if len(rewards) >= n:
            return_upto_prediction = multiplier.dot(rewards[-n:])
            G =  return_upto_prediction + (gamma**n)*(np.max(model.predict(observation)[0]))
            model.update(states[-n], actions[-n], G)

        total_reward += reward
        iters += 1

    #empty the cache
    rewards = rewards[n-1: ]
    actions = actions[n-1: ]
    states = states[n-1: ]

    #our game ends at 200th episode, so last few states don't have a chance to get updated, so tweaking a code little bit
    if observation[0] >= 0.5: #observed(0.5) from documentation
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            actions.pop(0)
            states.pop(0)
    else:
        #we did not make it to the goal
        while len(rewards) > 0:
            guess_rewards = rewards + [-1] * (n - len(rewards))
            G = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    return total_reward


if __name__ == "__main__":
    if "monitor" in sys.argv:
        qlearning_with_mountain_car.main(monitor = True)
    else:
        qlearning_with_mountain_car.main()





