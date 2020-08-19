# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:50:08 2020

Author: David O'Callaghan
"""

from gym_mo.envs.gridworlds.mo_gathering_env import MOGatheringEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense

from collections import deque # Used for replay buffer and reward tracking
from datetime import datetime # Used for timing script
import time


class ReplayMemory(deque):
    """
    Inherits from the 'deque' class to add a method called 'sample' for 
    sampling batches from the deque.
    """
    def sample(self, batch_size):
        """
        Sample a minibatch from the replay buffer.
        """
        # Random sample of indices
        indices = np.random.randint(len(self), 
                                    size=batch_size)
        # Filter the batch from the deque
        batch = [self[index] for index in indices]
        # Unpach and create numpy arrays for each element type in the batch
        states, actions, rewards, next_states, dones, weightss = [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(6)]
        return states, actions, rewards, next_states, dones, weightss


class RewardTracker:
    """
    Class for tracking mean rewards and storing all episode rewards for
    analysis.
    """
    def __init__(self, maxlen):
        self.moving_average = deque([-np.inf for _ in range(maxlen)], 
                                    maxlen=maxlen)
        self.maxlen = maxlen
        self.epsiode_rewards = []
        
    def __repr__(self):
        # For printing
        return self.moving_average.__repr__()
        
    def append(self, reward):
        self.moving_average.append(reward)
        self.epsiode_rewards.append(reward)
        
    def mean(self):
        return sum(self.moving_average) / self.maxlen
    
    def get_reward_data(self):
        episodes = np.array(
            [i for i in range(len(self.epsiode_rewards))]).reshape(-1,1)
        
        rewards = np.array(self.epsiode_rewards).reshape(-1,1)
        return np.concatenate((episodes, rewards), axis=1)


class PreferenceSpace:
    def sample(self):
        # Each preference weight is randomly sampled between -20 and 20 in steps of 5
        p0 = random.choice([x for x in range(-20, 21) if x % 5 == 0]) # Green
        p1 = random.choice([x for x in range(-20, 21) if x % 5 == 0]) # Red
        p2 = random.choice([x for x in range(-20, 21) if x % 5 == 0]) # Yellow
        p3 = random.choice([x for x in range(-20, 21) if x % 5 == 0]) # Other Agent Red
        pref = np.array([p0, p1, p2, p3], dtype=np.float32)
        w01 = np.array([-1, -5], dtype=np.float32)
        return np.concatenate((w01, pref))

    # # Uncomment below as necessary for training fixed agents
    # def sample(self):
    #     return np.array([-1, -5, +10, +20, +10, -20], dtype=np.float32) # Competitive
    #     # return np.array([-1, -5, +10, +20, +10, +20], dtype=np.float32) # Cooperative
    #     # return np.array([-1, -5, +20, +15, +20, +20], dtype=np.float32) # Fair
    #     # return np.array([-1, -5, +20,   0, +20, +20], dtype=np.float32) # Generous


class MovingAverage(deque):
    def mean(self):
        return sum(self) / len(self)


SEED = 42
DEBUG = False

BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 6_000

GAMMA = 0.99
ALPHA = 1e-4

TRAINING_EPISODES = 200_000
EXPLORATION_RESTARTS = 0

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (100_000 * 0.5)

COPY_TO_TARGET_EVERY = 1000 # Steps
START_TRAINING_AFTER = 50 # Episodes
MEAN_REWARD_EVERY = 100 # Episodes

FRAME_STACK_SIZE = 3

PATH_ID = 'tunable_DDMMYY'
PATH_DIR = './'

KALLSTROM_SETUP = False # Change to False for unit vector scalarisation

NUM_WEIGHTS = 4


class DQNAgent:
    
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(self.env.action_space.n)] 
        
        self.gamma = GAMMA # Discount
        self.eps0 = 1.0 # Epsilon greedy init
        
        self.batch_size = BATCH_SIZE
        self.replay_memory = ReplayMemory(maxlen=REPLAY_MEMORY_SIZE)
        self.reward_tracker = RewardTracker(maxlen=MEAN_REWARD_EVERY)
        
        image_size = self.env.observation_space.shape
        self.input_size = (*image_size[:2],image_size[-1]*FRAME_STACK_SIZE)
        self.output_size = self.env.action_space.n
        
        # Build both models
        self.model = self.build_model()
        self.target_model = self.build_model()
        # Make weights the same
        self.target_model.set_weights(self.model.get_weights())
        
        self.learning_plot_initialised = False
        self.env_penalty_sign = env.penalty_sign
        
    def build_model(self):
        """
        Construct the DQN model.
        """
        # image of size 8x8 with 3 channels (RGB)
        image_input = Input(shape=self.input_size)
        # preference weights
        weights_input = Input(shape=(NUM_WEIGHTS,)) # 6 weights

        # Define Layers
        x = image_input
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Concatenate()([x, weights_input])
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(self.env.action_space.n)(x)
        outputs = x
        
        # Build full model
        model = keras.Model(inputs=[image_input, weights_input], outputs=outputs)
        
        # Define optimizer and loss function
        self.optimizer = keras.optimizers.Adam(lr=ALPHA)
        self.loss_fn = keras.losses.mean_squared_error
        
        return model
    
    def normalise(self, weights):
        return weights / 40
    
    def get_scalarisation_weights(self, preference):
        # Step and Wall penalties should always have same weighting
        w0 = 0.025 # step_penalty
        w1 = 5*w0 # wall_penalty
        # All weights have to sum to 1
        max_sum = 1 - (w0+w1)
        # Make all positive
        w_rest = np.abs(preference[2:])
        # Normalise remaining weights to sum to max_sum
        if np.sum(w_rest) != 0:
            w_rest = max_sum * w_rest / np.sum(w_rest)
        else:
            w_rest = max_sum * (w_rest+1) / np.sum((w_rest+1))
        # Create full vector
        scalarisation_weights = np.array([w0, w1, *w_rest], dtype=np.float32)
        return scalarisation_weights
    
    def epsilon_greedy_policy(self, state, epsilon, weights):
        """
        Select greedy action from model output based on current state with 
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model.predict([state[np.newaxis], weights[np.newaxis]])
            return np.argmax(Q_values)
    
    def play_one_step(self, state, epsilon, preference):
        """
        Play one action using the DQN and store S A R S' in replay buffer.
        Adapted from: 
            https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
            [Accessed: 15/06/2020]
        """
        weights = self.normalise(preference[2:])
        if DEBUG:
            print(f'Preference: {preference}')
            print(f'Weights: {weights}')
        action = self.epsilon_greedy_policy(state, epsilon, weights)
        
        next_state, rewards, done, _ = self.env.step(action)
        next_state = np.float32(next_state) / 255 # convert to float32 for tf
        # Add frame to the stack of 3
        self.frame_stack.append(next_state)
        next_state = np.concatenate(self.frame_stack, axis=2)
        
        # Make preference positive and sum to 1
        if not self.env_penalty_sign:
            reward = np.dot(rewards, preference) # Linear scalarisation
        else:
            scalarisation_weights = self.get_scalarisation_weights(preference)
            reward = np.dot(rewards, scalarisation_weights) # Linear scalarisation
        self.replay_memory.append((state, action, reward, next_state, done, weights))
        return next_state, reward, done
    
    def training_step(self):
        """
        Train the DQN on a batch from the replay buffer.
        Adapted from: 
            https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
            [Accessed: 15/06/2020]
        """
        # Sample a batch of S A R S' from replay memory
        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, weightss = experiences
        
        # Compute target Q values from 'next_states'
        next_Q_values = self.target_model.predict([next_states, weightss])
        
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                       (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1) # Make column vector
        
        # Mask to only consider action taken
        mask = tf.one_hot(actions, self.output_size) # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self.model([states, weightss])
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, 
                                     keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, 
                                           self.model.trainable_variables))
        
    def train_model(self, episodes, pref_space):
        """
        Train the network over a range of episodes.
        """
        best_reward = -np.inf
        steps = 0
        
        for episode in range(1, episodes+1):
            preference = pref_space.sample()
            # Decay epsilon
            eps = max(self.eps0 - episode * EPSILON_DECAY, EPSILON_END)
            
            # Reset env
            state = self.env.reset(preference=preference)
            state = np.float32(state) / 255 # Convert to float32 for tf
            
            # Create deque for storing stack of N frames
            initial_stack = [state for _ in range(FRAME_STACK_SIZE)]
            self.frame_stack = deque(initial_stack, maxlen=FRAME_STACK_SIZE)
            state = np.concatenate(self.frame_stack, axis=2) # State is now a stack of frames
            
            episode_reward = 0
            while True:
                
                #eps = self.eps0
                state, reward, done = self.play_one_step(state, eps, preference)
                steps += 1
                episode_reward += reward
                if done:
                    break
                
                # Copy weights from main model to target model
                if steps % COPY_TO_TARGET_EVERY == 0:
                    if DEBUG:
                        print(f'\n\n{steps}: Copying to target\n\n')
                    self.target_model.set_weights(self.model.get_weights())
                        
            self.reward_tracker.append(episode_reward)
            avg_reward = self.reward_tracker.mean()
            if avg_reward > best_reward:
                best_reward = avg_reward
            
            print("\rTime: {}, Episode: {}, Reward: {}, Avg Reward {}, eps: {:.3f}".format(
                datetime.now() - start_time, episode, episode_reward, avg_reward, eps), end="")
            
            if episode > START_TRAINING_AFTER: # Wait for buffer to fill up a bit
                self.training_step()
                
            if episode % 500 == 0:
                self.model.save(MODEL_PATH)
                self.plot_learning_curve(image_path=IMAGE_PATH, 
                                         csv_path=CSV_PATH)
                
        self.model.save(MODEL_PATH)
        
    def load_model(self, path):
        self.model = keras.models.load_model(path)
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
            
    def plot_learning_curve(self, image_path=None, csv_path=None):
        """
        Plot the rewards per episode collected during training
        """
            
        colour_palette = get_cmap(name='Set1').colors
        if self.learning_plot_initialised == False:
            self.fig, self.ax = plt.subplots()
            self.learning_plot_initialised = True
        self.ax.clear()
        
        reward_data = self.reward_tracker.get_reward_data()
        x = reward_data[:,0]
        y = reward_data[:,1]
        
        # Save raw reward data
        if csv_path:
            np.savetxt(csv_path, reward_data, delimiter=",")
        
        # Compute moving average
        tracker = MovingAverage(maxlen=MEAN_REWARD_EVERY)
        mean_rewards = np.zeros(len(reward_data))
        for i, (_, reward) in enumerate(reward_data):
            tracker.append(reward)
            mean_rewards[i] = tracker.mean()
        
        # Create plot
        self.ax.plot(x, y, alpha=0.2, c=colour_palette[0])
        self.ax.plot(x[MEAN_REWARD_EVERY//2:], mean_rewards[MEAN_REWARD_EVERY//2:], 
                c=colour_palette[0])
        self.ax.set_xlabel('episode')
        self.ax.set_ylabel('reward per episode')
        self.ax.grid(True, ls=':')
        
        # Save plot
        if image_path:
            self.fig.savefig(image_path)
    
    def play_episode(self, preference):
        """
        Play one episode using the DQN and display the grid image at each step.
        """
        state = self.env.reset(preference=preference)
        state = np.float32(state) / 255 # Convert to float32 for tf
        weights = self.normalise(preference[2:])
        
        # Create stack
        initial_stack = [state for _ in range(FRAME_STACK_SIZE)]
        self.frame_stack = deque(initial_stack, maxlen=FRAME_STACK_SIZE)
        state = np.concatenate(self.frame_stack, axis=2)
    
        print("Initial State:")
        self.env.render()
        i = 0
        episode_reward = 0
        while True:
            i += 1
            action = self.epsilon_greedy_policy(state, 0.05, weights)
            print('Move #: %s; Taking action: %s' % (i, action))
            state, rewards, done, _ = self.env.step(action)
            state = np.float32(state) / 255 # convert to float32 for tf
            # Add to stack
            self.frame_stack.append(state)
            state = np.concatenate(self.frame_stack, axis=2)
            # Scalarise reward vector
            if not self.env_penalty_sign:
                reward = np.dot(rewards, preference) # Linear scalarisation
            else:
                scalarisation_weights = self.get_scalarisation_weights(preference)
                reward = np.dot(rewards, scalarisation_weights) # Linear scalarisation
            episode_reward += reward
            self.env.render()
            time.sleep(0.1)
            if done:
                print(f'Reward: {episode_reward}')
                break


fixed_preferences_dict = {
    '1': np.array([-1, -5, +10, +20, +10, -20]), # Competitive
    '2': np.array([-1, -5, +10, +20, +10, +20]), # Cooperative
    '3': np.array([-1, -5, +20, +15, +20, +20]), # Fair
    '4': np.array([-1, -5, +20,   0, +20, +20]), # Generous
    }

if __name__ == '__main__':
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    
    print(tf.__version__)
    try:
        print(tf.config.experimental_list_devices())
    except:
        print(tf.config.list_physical_devices())
        
    IMAGE_PATH = f'{PATH_DIR}/plots/reward_plot_{PATH_ID}.png'
    CSV_PATH = f'{PATH_DIR}/plots/reward_data_{PATH_ID}.csv'
    MODEL_PATH = f'{PATH_DIR}/models/dqn_model_{PATH_ID}.h5'
    
    # For timing the script
    start_time = datetime.now()
    
    # Instantiate environment
    if KALLSTROM_SETUP:
        item_env = MOGatheringEnv(penalty_sign=False)
    else:
        item_env = MOGatheringEnv(penalty_sign=True)
    
    # Initialise Preference Weight Space
    pref_space = PreferenceSpace()
    
    # Instantiate agent (pass in environment)
    dqn_ag = DQNAgent(item_env)
    # Uncomment as necessary to load a pre-trained model
    #dqn_ag.load_model('models/dqn_model_tunable_DDMMYY.h5')

    dqn_ag.train_model(TRAINING_EPISODES, pref_space)
    # To re-initialise exploration (not actually used)
    for _ in range(EXPLORATION_RESTARTS):
        dqn_ag.train_model(TRAINING_EPISODES, pref_space)
    
    # Plot learning curve
    dqn_ag.plot_learning_curve(image_path=IMAGE_PATH, 
                               csv_path=CSV_PATH)
    
    # Play episode with learned DQN
    pref = pref_space.sample()
    dqn_ag.play_episode(preference=pref)
        
    run_time = datetime.now() - start_time
    print(f'Run time: {run_time} s')
