import gym
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.activations import softmax, tanh
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from gym import wrappers
from datetime import datetime
import numpy as np
from qlearning_with_mountain_car import plot_running_avg, FeatureTransformer
tf.keras.backend.set_floatx('float64')

def cost_for_policy_gradient(advantages, norm, actions_taken_list):
    #contionous
    log_probs = norm.log_prob(actions_taken_list)
    cost = -tf.reduce_sum(input_tensor=advantages * log_probs + 0.1*norm.entropy())
    return cost


#for testing different architecture
class HiddenLayer(layers.Layer):
    def __init__(self, input_dimension, hidden_units, f, use_bias=True, zeros=False):
        super().__init__()
        if zeros:
            self.w = self.add_weight(shape=(input_dimension, hidden_units), initializer="random_normal", trainable=True)
        else:
            self.w = self.add_weight(shape=(input_dimension, hidden_units), initializer="zeros", trainable=True)

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

class PolicyModel(layers.Layer):
    def __init__(self, D, ft, hidden_layer_sizes):
        super().__init__()
        #save inputs for copy
        self.ft =ft
        self.D = D
        self.hidden_layer_sizes = hidden_layer_sizes

        self.sequence = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2, tanh)
            self.sequence.append(layer)
            M1 = M2

        self.mean_layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
        self.std_layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)
        self.cost = cost_for_policy_gradient
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.norm_dist = tfp.distributions.Normal

    def call(self, X):
        for layer in self.sequence:
            X  = layer(X)
        mean = tf.reshape(self.mean_layer(X), [-1])
        std = tf.reshape(self.std_layer(X), [-1]) + 1e-5
        norm = self.norm_dist(mean, std)
        return norm, tf.clip_by_value(norm.sample(), -1, 1)

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        with tf.GradientTape() as tape:
            norm, p_a_given_s = self(X)
            loss_value = self.cost(advantages, norm, actions)
        #print(self.trainable_weights)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
  
    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        _, p_a_given_s = self(X)
        return p_a_given_s
  
    def sample_action(self, X):
        p = self.predict(X)[0]
        return p


class ValueModel(layers.Layer):
    def __init__(self, D, ft, hidden_layer_sizes=[]):
        super().__init__()
        self.ft = ft
        self.nn_layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            nn_layer = HiddenLayer(M1, M2, tanh)
            self.nn_layers.append(nn_layer)
            M1 = M2
        #final layer
        final_layer = HiddenLayer(M1, 1, lambda x: x)
        self.nn_layers.append(final_layer)
    
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=10e-5)
        self.loss_fn = tf.keras.losses.MSE

    def call(self, X):
        for nn_layer in self.nn_layers:
            X  = nn_layer(X)
        return X

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        X = self.ft.transform(X)
        with tf.GradientTape() as tape:
            logits = self(X)  # Logits for this minibatch
            # Loss value for this minibatch
            loss_value = sum(self.loss_fn(Y, logits))
        #print(self.trainable_weights)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self(X)


def play_one_td(pmodel, vmodel, gamma, env):
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0

    while not done:
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step([action])
        total_reward+=reward 
        G = reward + gamma*(vmodel.predict(observation))
        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)
           
        iters += 1
    
    return total_reward, iters


def main(monitor=False):
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env, n_components=1000)
    D = ft.dimensions
    pmodel = PolicyModel(D, ft, [])
    vmodel = ValueModel(D, ft, []) 
    gamma = 0.95
    

    if monitor:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 50
    total_rewards = np.empty(N)
    for n in range(N):
        total_reward, num_steps = play_one_td(pmodel, vmodel, gamma, env)
        total_rewards[n] = total_reward
        if n % 1 == 0:
            print("episode:", n, "total reward: %.1f" % total_reward, "num steps: %d" % num_steps, "avg reward (last 100): %.1f" % total_rewards[max(0, n-100):(n+1)].mean(), "avg reward (first 100): %.1f" % total_rewards[0:100].mean())
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
