import gym
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.activations import softmax, tanh
from matplotlib import pyplot as plt
from gym import wrappers
from datetime import datetime
from qlearning_with_bins import plot_running_avg
tf.keras.backend.set_floatx('float64')

def cost_for_policy_gradient(advantages, p_a_given_s, actions_taken_list, num_of_actions):
    selected_probs = tf.math.log(tf.reduce_sum(p_a_given_s * tf.one_hot(actions_taken_list, num_of_actions, dtype='float64'), axis=1))
    return -tf.reduce_sum(advantages * selected_probs)


#for testing different architecture
class HiddenLayer(layers.Layer):
    def __init__(self, input_dimension, hidden_units, f, use_bias=True):
        super().__init__()
        self.w = self.add_weight(shape=(input_dimension, hidden_units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(hidden_units, ), initializer="random_normal", trainable=True)
        self.use_bias = use_bias
        self.f = f 

    def call(self, input):
        if self.use_bias:
            z = tf.matmul(input, self.w) + self.b
        else:
            z = tf.matmul(input, self.w)
        return self.f(z)

class PolicyModel(layers.Layer):
    def __init__(self, D, K, hidden_layer_sizes):
        super().__init__()
        #K - Number of actions
        self.K = K
        self.seq_model = Sequential()
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2, tanh)
            self.seq_model.add(layer)
            M1 = M2
        #final layer
        #final_layer = HiddenLayer(M1, K, softmax, use_bias=False)
        self.seq_model.add(tf.keras.layers.Dense(K, activation='softmax'))
        self.cost = cost_for_policy_gradient
        self.optimizer = tf.keras.optimizers.Adagrad(10e-2)

    def call(self, X):
        return self.seq_model(X)

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        with tf.GradientTape() as tape:
            p_a_given_s = self(X)  # Logits for this minibatch
            # Loss value for this minibatch
            loss_value = self.cost(advantages, p_a_given_s, actions, self.K)
        #print(self.trainable_weights)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
    def predict(self, X):
        X = np.atleast_2d(X)
        return self(X)

    def sample_action(self, X):
        p = self.predict(X)[0]
        return np.random.choice(len(p), p=p)
    
class ValueModel(layers.Layer):
    def __init__(self, D, hidden_layer_sizes):
        super().__init__()
        #K - Number of actions
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
        with tf.GradientTape() as tape:
            logits = self(X)  # Logits for this minibatch
            # Loss value for this minibatch
            loss_value = sum(self.loss_fn(Y, logits))
        #print(self.trainable_weights)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
    def predict(self, X):
        X = np.atleast_2d(X)
        return self(X)


def play_one_td(pmodel, vmodel, gamma, env):
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0

    while not done:
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        if done and iters < 199:
            reward = -200

        G = reward + vmodel.predict(observation)[0]
        advantage = G - vmodel.predict(prev_observation)[0]
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)
        
        if reward == 1:
            total_reward+=reward    
        iters += 1
    
    return total_reward
        


def play_one_mc(pmodel, vmodel, gamma, env):
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0

    states = []
    actions = []
    rewards = []

    while not done:
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        if done and iters < 199:
            reward = -200

        states.append(prev_observation)
        rewards.append(reward)
        actions.append(action)

        if reward == 1:
            total_reward+=reward    
        iters += 1

    returns = []
    advantages = []
    G = 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G - vmodel.predict(s)[0])
        G = r + gamma * G

    returns.reverse()
    advantages.reverse()
    pmodel.partial_fit(states, actions, advantages)
    vmodel.partial_fit(states, returns)

    return total_reward


def main(monitor=False):
    env = gym.make('CartPole-v0')
    D = env.observation_space.shape[0]
    K = env.action_space.n
    pmodel = PolicyModel(D, K, [])
    vmodel = ValueModel(D, [10])
    gamma = 0.99

    
    if monitor:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 2000
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 1/np.sqrt(n+1)
        total_reward = play_one_mc(pmodel, vmodel, gamma, env)
        total_rewards[n] = total_reward
        if n % 1 == 0:
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


    