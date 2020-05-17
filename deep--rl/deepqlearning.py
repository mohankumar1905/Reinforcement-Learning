import gym
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from gym import wrappers
from datetime import datetime
from qlearning_with_bins import plot_running_avg
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.activations import softmax, tanh
tf.keras.backend.set_floatx('float64')

def cost(G, Y_hat, actions_taken_list, num_of_actions):
    selected_action_values = tf.reduce_sum(Y_hat * tf.one_hot(actions_taken_list, num_of_actions, dtype='float64'), axis=1)
    return tf.reduce_sum(tf.math.square(G - selected_action_values))


class HiddenLayer(layers.Layer):
    def __init__(self, input_dimension, hidden_units, f, use_bias=True):
        super().__init__()
        self.w = self.add_weight(shape=(input_dimension, hidden_units), initializer="random_normal", trainable=True)
        if use_bias:
            self.b = self.add_weight(shape=(hidden_units, ), initializer="zeros", trainable=True)
        self.use_bias = use_bias
        self.f = f  

    def call(self, input):
        if self.use_bias:
            z = tf.matmul(input, self.w) + self.b
        else:
            z = tf.matmul(input, self.w)
        return self.f(z)


class DQN(layers.Layer):
    def __init__(self, D, K, hidden_layer_sizes, gamma, max_experience=10000, min_experience=100, batch_size=32):
        super().__init__()
        #K - Number of actions
        self.K = K
        self.D = D
        self.hidden_layer_sizes = hidden_layer_sizes
        #For creating replay memory
        self.max_experience = max_experience
        self.min_experience = min_experience
        self.batch_size = batch_size
        self.gamma = gamma
        self.experience = {'s': [], 'a':[], 'r':[], 's2':[], 'done':[]}
        self.sequence = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2, tanh)
            self.sequence.append(layer)
            M1 = M2
        layer = HiddenLayer(M1, K, lambda x: x)
        self.sequence.append(layer)
        self.cost = cost
        self.optimizer = tf.keras.optimizers.Adam(10e-4)


    def call(self, X):
        for layer in self.sequence:
            X  = layer(X) 
        return X       

    def partial_fit(self, X, G, actions):
        X = np.atleast_2d(X)
        G = np.atleast_1d(G)
        with tf.GradientTape() as tape:
            Y_hat = self(X)  
            # Loss value for this minibatch
            loss_value = cost(G, Y_hat, actions, self.K)
        #print(self.trainable_weights)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    def copy_from(self, other):
        weights = other.get_weights()
        self.set_weights(weights)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self(X)

    def train(self, target_network):
        #sample a random batch from the buffer and do an iteration of gradient descent
        if len(self.experience['s']) < self.min_experience:
            #don't do any thing if we don't have enough experience.
            return
        #randomly select a batch 
        idx = np.random.choice(len(self.experience['s']), size=self.batch_size, replace=False)
        #print(idx)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = [r+self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        self.partial_fit(states, targets, actions)

    def add_experience(self, s, a, r, s2, done):
        if len(self.experience['s']) >= self.max_experience:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)

    def sample_action(self, X, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(X)
            return np.argmax(self.predict(X)[0])


def play_one(model, tmodel, eps, gamma, copy_period, env):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done and iters < 199:
            reward = -300

        #update the model
        model.add_experience(prev_observation, action, reward, observation, done)
        model.train(tmodel)
        

        iters += 1
        if iters % copy_period == 0:
            tmodel.copy_from(model)
    return totalreward


def main(monitor=False):
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_period = 50
    D = len(env.observation_space.sample())
    K = env.action_space.n
    sizes = [2000, 2000]
    model = DQN(D, K, sizes, gamma)
    tmodel = DQN(D, K, sizes, gamma)

    if monitor:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 500
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(model, tmodel, eps, gamma, copy_period, env)
        totalrewards[n] = totalreward
        if n % 1 == 0:
            print("episode: ", n, "total_reward: ", totalreward, "eps:", eps)
    print("Average reward over last 100 episodes: ", totalrewards[-100:].mean())
    print("total_steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    plot_running_avg(totalrewards)

if __name__ == "__main__":
    if "monitor" in sys.argv:
        main(monitor = True)
    else:
        main()