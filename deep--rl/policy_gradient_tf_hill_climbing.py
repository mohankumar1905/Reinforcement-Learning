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
  def __init__(self, ft, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_variance=[]):
    super().__init__()
    #save inputs for copy
    self.ft =ft
    self.D = D
    self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
    self.hidden_layer_sizes_variance = hidden_layer_sizes_variance

    self.mean_sequence = []
    M1 = D
    for M2 in hidden_layer_sizes_mean:
        layer = HiddenLayer(M1, M2, tanh)
        self.mean_sequence.append(layer)
        M1 = M2

    layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
    self.mean_sequence.append(layer)


    self.varaince_sequence = []
    M1 = D
    for M2 in hidden_layer_sizes_variance:
        layer = HiddenLayer(M1, M2, tanh)
        self.varaince_sequence.append(layer)
        M1 = M2

    layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)
    self.varaince_sequence.append(layer)

    #gather parameters
    self.all_layers = []
    for layer in (self.mean_sequence + self.varaince_sequence):
      self.all_layers.append(layer)

  def call(self, X):
    def get_ouptut(X, sequence):
      for layer in sequence:
        X  = layer(X)
      return tf.reshape(X, [-1])

    mean = get_ouptut(X, self.mean_sequence)
    std = get_ouptut(X, self.varaince_sequence) + 1e-4 #smoothing
    norm = tfp.distributions.Normal(mean, std)
    return tf.clip_by_value(norm.sample(), -1, 1)

  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    return self(X)
  
  def sample_action(self, X):
    p = self.predict(X)[0]
    return p

  def copy(self):
    clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_variance)
    weights = self.get_weights()
    clone.set_weights(weights)
    return clone

  """
  def copy_from(self, other_model):
    current_model_params = self.params
    other_model_params = other_model.params
    for (cml, otml) in zip(current_model_params, other_model_params):
      print(cml.shape)
      print(otml.shape)
      cml.assign(otml)
  """
  
  def preturb_params(self):
    for layer_index in range(len(self.all_layers)):
      #print("weights")
      #print(self.all_layers[layer_index].get_weights())
      new_weights = []
      for weight_index in range(len(self.all_layers[layer_index].weights)):
        weight = self.all_layers[layer_index].weights[weight_index]
        if len(weight.shape) == 2:
          noise = np.random.randn(weight.shape[0], weight.shape[1]) / np.sqrt(weight.shape[0]) * 5.0
        elif len(weight.shape) == 1:
          noise = np.random.randn(weight.shape[0]) / np.sqrt(weight.shape[0]) * 5.0
        else:
          raise
          #print("written for only two dimensions")
        if np.random.random() < 0.1:
          #print("assign_noise")
          new_weights.append(noise)
        else:
          #print("add_noise")
          new_weights.append(weight+noise)
        #print("weight", weight)
        #print("noise", noise)
        #print("new_weights", new_weights)
      self.all_layers[layer_index].set_weights(new_weights) 
      #print("updated_weight", self.all_layers[layer_index].get_weights())
      #raise


def play_one(env, pmodel, gamma):
  observation = env.reset()
  done = False
  total_reward = 0
  iters = 0

  while not done:
    action = pmodel.sample_action(observation)
    # oddly, the mountain car environment requires the action to be in
    # an object where the actual action is stored in object[0]
    observation, reward, done, info = env.step([action])
    total_reward += reward

    iters += 1

  return total_reward

def play_multiple_episodes(env, num_episodes, pmodel, gamma, print_iters=False):
  total_rewards = np.empty(num_episodes)
  for i in range(num_episodes):
    total_rewards[i] = play_one(env, pmodel, gamma)
    if print_iters:
      print(i, "averge so far", total_rewards[: (i+1)].mean())
    
  avg_total_rewards = total_rewards.mean()
  print("average total rewards:", avg_total_rewards)
  return avg_total_rewards

def random_hill_climb_weight_search(env, pmodel, gamma):
  total_rewards = []
  best_avg_totalreward = float('-inf')
  best_pmodel = pmodel
  num_episodes_per_params_test = 3

  for t in range(100):
    temp_pmodel = best_pmodel.copy()
    temp_pmodel.preturb_params()
    avg_total_rewards = play_multiple_episodes(env, num_episodes_per_params_test, temp_pmodel, gamma)
    total_rewards.append(avg_total_rewards)
    if avg_total_rewards > best_avg_totalreward:
      best_avg_totalreward = avg_total_rewards
      best_pmodel = temp_pmodel
  return total_rewards, best_pmodel

def main(monitor=False):
  env = gym.make('MountainCarContinuous-v0')
  ft = FeatureTransformer(env, n_components=3)
  D = ft.dimensions
  pmodel = PolicyModel(ft, D, [], [])
  gamma = 0.99

  if monitor:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

  total_rewards, pmodel = random_hill_climb_weight_search(env, pmodel, gamma)
  print("Max Reward", np.max(total_rewards))

  #play 100 episodes and check the average
  avg_total_rewards = play_multiple_episodes(env, 100, pmodel, gamma, print_iters=True)
  print("avg reward over 100 episodes with best models:", avg_total_rewards)

  plt.plot(total_rewards)
  plt.title("Rewards")
  plt.show()


if __name__ == "__main__":
    if "monitor" in sys.argv:
        main(monitor = True)
    else:
        main()