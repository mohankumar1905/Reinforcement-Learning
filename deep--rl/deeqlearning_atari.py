import copy
import gym
import os
import sys
import random
import numpy as np 
from matplotlib import pyplot as plt
from gym import wrappers
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.activations import softmax, tanh
tf.keras.backend.set_floatx('float64')

MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 80
K = 4 #env.action_space.n

#Transfrom raw images for input into neural network
class ImageTransformer:
    '''Converts the image to grayscale and crops it to IM_SIZE, IM_SIZE'''
    def __init__(self, offset_height=34, offset_width=0, target_height=160, target_width=160):
        self.offset_height = offset_height
        self.offset_width = offset_width
        self.target_height = target_height
        self.target_width = target_width

    def transfrom(self, state):
        output = tf.image.rgb_to_grayscale(state)
        output = tf.image.crop_to_bounding_box(output, self.offset_height, self.offset_width, self.target_height, self.target_width)
        output = tf.image.resize(output, [IM_SIZE, IM_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.squeeze(output)

def update_state(current_state, next_frame):
    '''Takes in the current state, next frame and returns the next state'''
    return np.append(current_state[:,:,1:], np.expand_dims(next_frame, 2), axis=2)


class ReplayMemory:
    def __init__(self, size=MAX_EXPERIENCES, frame_height=IM_SIZE, frame_width=IM_SIZE, agent_history_length=4, batch_size=32):
        '''
        Basic Idea here is we are going to pre allocate all of the frames, we plan on storing and then we can sample states from 
        the individual states later on.

        Args
        size: (buffer_size): Integer, Number of Stored transitions
        frame_height: Height of frame of an Atari Game
        frame_width: Width of frame of an Atari Game
        agent_history_length: Integer, Number of frames stacked together to create a state. 
        batch_size: Integer, Number of transactions returned in a minibatch
        '''

        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        #both count and current keeps track of the insertion point in replay buffer.
        self.count = 0 
        self.current = 0

        #Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        #Pre-allocate memory for the states and new states in a minibatch.
        self.states = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width), dtype=np.uint8) 
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        '''
        Args:
            action: An integer encoded action
            frame: One grayscale image of the frame
            reward: reward that the agent recieved for performing the action
            terminal: A bool stating whether the episode terminated
        '''

        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of the frame is wrong!')
        
        self.actions[self.current] = action
        self.frames[self.current] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty")
        if index <  self.agent_history_length - 1:
            raise ValueError("Index must be minimum 3")
        return self.frames[index-self.agent_history_length+1 : index+1]

    
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current_state:
                    #checks for frames that is between old an new
                    continue
                if self.terminal_flags[index - self.agent_history_length : index].any():
                    #checks for done flag
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        '''Returns a minibatch of self.batch_size transitions'''
        
        if self.count < self.agent_history_length:
            raise ValueError("Not enough memories to get a mini batch")
        
        self._get_valid_indices()
        
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx-1)
            self.new_states[i] = self._get_state(idx)
        
        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], \
            self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices ]


def cost(G, Y_hat, actions_taken_list, num_of_actions):
    selected_action_values = tf.reduce_sum(Y_hat * tf.one_hot(actions_taken_list, num_of_actions, dtype='float64'), axis=1)
    #return tf.reduce_sum(tf.math.square(G - selected_action_values))
    return tf.reduce_sum(tf.compat.v1.losses.huber_loss (G, selected_action_values))

class DQN(Model):
    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes):
        super(DQN, self).__init__()
        self.K = K
        self.conv_seq = Sequential()
        for num_output_filters, filter_size, pool_size in conv_layer_sizes:
            self.conv_seq.add(layers.Conv2D(num_output_filters, filter_size, strides=pool_size, activation='relu', input_shape=(IM_SIZE, IM_SIZE, 4)))
        self.flatten = layers.Flatten()
        self.linear_sequence = Sequential()
        for M in hidden_layer_sizes:
            layer = layers.Dense(M) 
            self.linear_sequence.add(layer)
        layer = layers.Dense(K) 
        self.linear_sequence.add(layer)
        self.cost = cost
        self.optimizer = tf.keras.optimizers.Adam(10e-4)

    def call(self, X):
        Z = X/255
        Z = self.conv_seq(Z)
        Z = self.flatten(Z)
        Z = self.linear_sequence(Z)
        return Z

    def update(self, X, G, actions):
        with tf.GradientTape() as tape:
            Y_hat = self(X)  
            # Loss value for this minibatch
            loss_value = cost(G, Y_hat, actions, self.K)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value

    def copy_from(self, other):
        weights = other.get_weights()
        self.set_weights(weights)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self(X)

    def sample_action(self, X, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([X])[0])

    
def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    
    #sample experiences
    states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()
    
    #calculate targets
    next_Qs = target_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)
    targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q
    
    #update model
    loss = model.update(states, targets, actions)
    
    return loss



def play_one(env, total_t, experience_replay_buffer, model, target_model,
                 image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_min):
    '''
    playing episodes
    Arguments explained
    total_t - total number of steps played so far
    experience_replay_buffer - ReplayMemory object
    '''

    t0 = datetime.now()
    #Reset the environment
    obs = env.reset()
    obs_small = image_transformer.transfrom(obs)
    state = np.stack([obs_small] * 4, axis=2)

    loss=None
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    done=False

    while not done:
        #check if it is time to update target network and then update it.
        if (total_t % TARGET_UPDATE_PERIOD == 0 and total_t!=0) or total_t == 1:
            print("copying")
            target_model.copy_from(model)
            print ("parameters copied to target network. total steps: ", total_t, "target update period: ", TARGET_UPDATE_PERIOD)
    
        action = model.sample_action(state, epsilon)
        obs, reward, done, _ = env.step(action)
    
        obs_small = image_transformer.transfrom(obs)
        next_state = update_state(state, obs_small)

        episode_reward += reward
    
        #save the latest experience
        experience_replay_buffer.add_experience(action, obs_small, reward, done)

        #Train the Model, keep track of time
        t0_2 = datetime.now()
    
        loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
    
        dt = datetime.now() - t0_2

        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1

        state = next_state
        total_t += 1

        epsilon = max(epsilon - epsilon_change, epsilon_min)
    
    return total_t, episode_reward, (datetime.now()- t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon


def smooth(x):
    #last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i - start + 1)
    return y

def main():
    #hyperparmeters and initializations
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_size = 32
    num_episodes = 3500
    total_t = 0

    experience_replay_buffer = ReplayMemory()
    episode_rewards = np.zeros(num_episodes)

    epsilon_min = 0.1
    epsilon = 1.0
    epsilon_change = (epsilon - epsilon_min) / 500000

    #Create environment
    env = gym.envs.make("Breakout-v0")


    #Create Models
    model = DQN(K, conv_layer_sizes, hidden_layer_sizes)
    target_model = DQN(K, conv_layer_sizes, hidden_layer_sizes)

    image_transformer = ImageTransformer()


    print("Populating Experience Replay Buffer")
    obs = env.reset()

    for i in range(MIN_EXPERIENCES):
        action = np.random.choice(K)
        obs, reward, done, _ = env.step(action)
        obs_small = image_transformer.transfrom(obs)
        experience_replay_buffer.add_experience(action, obs_small, reward, done)
        if i % 5000 == 0:
            print(i, "random actions taken")
        if done:
            obs = env.reset()
    
    #Play a number of episodes and learn.
    print("start playing")
    t0 = datetime.now()
    for i in range(num_episodes):
        total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(env, total_t, experience_replay_buffer, model, target_model, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_min)
        episode_rewards[i] = episode_reward

        last_100_avg = episode_rewards[max(0, i-100) : i + 1].mean()

        print("Episode: ", i, "Duration: ", duration, "Num_steps: ", num_steps_in_episode, "Reward: ", episode_reward, "Training time per step: ", time_per_step, "Average_reward_last_100: ", last_100_avg, "Epsilon: ", epsilon)

    print("Total duration", datetime.now() - t0)
    model.save_weights('weights.h5')

    #plot the smoothed returns
    y =  smooth(episode_rewards)
    plt.plot(episode_rewards, label='org')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()