# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:19:28 2020

Author: David O'Callaghan
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import random
import pandas as pd

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


from gym_wolfpack_custom.envs import MOWolfpackCustomEnv

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
    
    # distribution = [
    #     [0.005, 0.025, 0.485, 0.485], # Even
    #     [0.005, 0.025, 0.150, 0.820], # Cooperative
    #     [0.005, 0.025, 0.820, 0.150], # Competitive
    #     ]
    # def sample(self):
    #     return np.array(random.choice(self.distribution), dtype=np.float32)
    
    # def sample(self):
    #     return np.array([0.005, 0.025, 2, 2], dtype=np.float32)
    def __init__(self):
        w0 = 0.005 # Time penalty
        w1 = 5 * w0 # Wall penalty : 5x time penalty
        w2_range = np.linspace(0,0.97,5)
        
        self.distribution = [np.array([w0, w1, w2, 0.97 - w2], dtype=np.float32) for w2 in w2_range]
        # self.prob_weighting = [0.6/4, 0.6/4, 0.6/4, 0.6/4, 0.4]
        
    def sample(self):
        # return np.random.choice(self.distribution, p=self.prob_weighting)
        return random.choice(self.distribution)
        
    
    # def sample(self):
    #     w0 = 0.005 # Time penalty
    #     w1 = 5 * w0 # Wall penalty : 5x time penalty
    #     max_reward = 1 - (w0 + w1)
    #     w2 = np.random.rand() * max_reward # Lone reward
    #     w3 = 1 - (w0 + w1 + w2) # Team reward
    #     return np.array([w0, w1, w2, w3], dtype=np.float32) 
    
    
class MovingAverage(deque):
    def mean(self):
        return sum(self) / len(self)


SEED = 42
DEBUG = False
IMAGE = True

BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 6000

GAMMA = 0.99
ALPHA = 1e-4

TRAINING_EPISODES = 25_000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (TRAINING_EPISODES * 0.15)

COPY_TO_TARGET_EVERY = 1000 # Steps
START_TRAINING_AFTER = 50 # Episodes
MEAN_REWARD_EVERY = 300 # Episodes

FRAME_STACK_SIZE = 3

#PATH_ID = '130720_5pref'
NUM_WEIGHTS = 2


class DQNAgent:
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.actions = [i for i in range(env.action_space.n)] 
        
        self.gamma = GAMMA # Discount
        self.eps0 = 1.0 # Epsilon greedy init
        
        self.batch_size = BATCH_SIZE
        self.replay_memory = ReplayMemory(maxlen=REPLAY_MEMORY_SIZE)
        self.reward_tracker = RewardTracker(maxlen=MEAN_REWARD_EVERY)
        
        if IMAGE:
            image_size = env.observation_space.shape
            self.input_size = (*image_size[:2],image_size[-1]*FRAME_STACK_SIZE)
        else:
            self.input_size = env.observation_space.shape

        self.output_size = env.action_space.n
        
        # Build both models
        self.model = self.build_model()
        self.target_model = self.build_model()
        # Make weights the same
        self.target_model.set_weights(self.model.get_weights())
        
        self.learning_plot_initialised = False
        
    def build_model(self):
        """
        Construct the DQN model.
        """
        if IMAGE:
            # image of size 8x8 with 3 channels (RGB)
            image_input = Input(shape=self.input_size)
            # preference weights
            weights_input = Input(shape=(NUM_WEIGHTS,))
    
            # Define Layers
            x = image_input
            x = Conv2D(256, (3, 3), activation='relu')(x)
            # x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.2)(x)
            x = Conv2D(256, (3, 3), activation='relu')(x)
            # x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.2)(x)
            x = Flatten()(x)
            x = Concatenate()([x, weights_input])
            x = Dense(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            
            x = Dense(env.action_space.n)(x)
            outputs = x
            
            # Build full model
            model = keras.Model(inputs=[image_input, weights_input], outputs=outputs)
            # model = keras.Model(inputs=image_input, outputs=outputs)
            
            # Define optimizer and loss function
            self.optimizer = keras.optimizers.Adam(lr=ALPHA)
            #self.loss_fn = keras.losses.mean_squared_error
            self.loss_fn = keras.losses.Huber()
        else:
            # agent locations and distances between them
            state_input = Input(shape=self.input_size)
            # preference weights
            weights_input = Input(shape=(NUM_WEIGHTS,))
    
            # Define Layers
            x = Concatenate()([state_input, weights_input])
            x = Dense(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            x = Dense(env.action_space.n)(x)
            outputs = x
            
            # Build full model
            model = keras.Model(inputs=[state_input, weights_input], outputs=outputs)
            
            # Define optimizer and loss function
            self.optimizer = keras.optimizers.Adam(lr=ALPHA)
            #self.loss_fn = keras.losses.mean_squared_error
            self.loss_fn = keras.losses.Huber()
        
        return model    
    
    def epsilon_greedy_policy(self, state, epsilon, weights):
    # def epsilon_greedy_policy(self, state, epsilon):
        """
        Select greedy action from model output based on current state with 
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model.predict([state[np.newaxis], weights[np.newaxis]])
            # Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values)
    
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
        # states, actions, rewards, next_states, dones = experiences
        
        # Compute target Q values from 'next_states'
        next_Q_values = self.target_model.predict([next_states, weightss])
        # next_Q_values = self.target_model.predict(next_states)
        
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                       (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1) # Make column vector
        
        # Mask to only consider action taken
        mask = tf.one_hot(actions, self.output_size) # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self.model([states, weightss])
            # all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, 
                                     keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, 
                                           self.model.trainable_variables))
    
    def load_model(self, path):
        self.model = keras.models.load_model(path)
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
    
    def update_target_model(self):
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
    

if __name__ == '__main__':
    
    PATH_DIR = './models/'
    EPISODES = 250

    print(tf.__version__)
    try:
        print(tf.config.experimental_list_devices())
    except:
        print(tf.config.list_physical_devices())
    
    N_PREDATOR = 3
    env = MOWolfpackCustomEnv(n_predator=N_PREDATOR, image=IMAGE)
    
    # For timing the script
    start_time = datetime.now()
    
    random.seed(SEED)
    np.random.seed(SEED)
    env.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Initialise agents
    # prey1 = DQNAgent(0)
    pred1 = DQNAgent(1)
    pred2 = DQNAgent(2)
    pred3 = DQNAgent(3)    
    
    pred1.load_model(f'{PATH_DIR}/wolfpack_model_tunable_pred1_seed1.h5')
    pred2.load_model(f'{PATH_DIR}/wolfpack_model_tunable_pred2_seed1.h5')
    pred3.load_model(f'{PATH_DIR}/wolfpack_model_tunable_pred1_seed2.h5')
    
    
    
    if not IMAGE:
        x, y = env.base_gridmap_array.shape[0] - 1, env.base_gridmap_array.shape[1] - 1
        max_state = np.array([x, y, *[z for _ in range(2) for z in [x, y, x+y]]], dtype=np.float32)

    steps = 0
    
    pref_space = PreferenceSpace()
    
    prefs = np.linspace(0, 0.97, 9)
    #prefs = np.linspace(0.12125, 0.84875, 4)
    results = []
    for pref in prefs:
        
        steps = 0
    
        lone1_captures = 0
        lone2_captures = 0
        lone3_captures = 0
        
        pair_captures = 0
        team_captures = 0

        pref1 = np.array([0.005, 0.025, pref, 0.97 - pref], dtype=np.float32)
        pref2 = pref1.copy()
        pref3 = pref1.copy()
        
        weights1 = pref1[2:]
        weights2 = pref2[2:]
        weights3 = pref3[2:]
        
        print(f'\n\n{np.round(pref1[2:], 3)}\n-----------------')
        
        for episode in range(1, EPISODES+1):
            # Decay epsilon
            eps = 0.01
            
            # Reset env
            observations = env.reset()
            #env.render()
            prey_state, pred1_state, pred2_state, pred3_state = observations
            
            if IMAGE:
                # Create deque for storing stack of N frames
                # Pred 1
                pred1_initial_stack = [pred1_state for _ in range(FRAME_STACK_SIZE)]
                pred1_frame_stack = deque(pred1_initial_stack, maxlen=FRAME_STACK_SIZE)
                pred1_state = np.concatenate(pred1_frame_stack, axis=2) # State is now a stack of frames
                # Pred 2
                pred2_initial_stack = [pred2_state for _ in range(FRAME_STACK_SIZE)]
                pred2_frame_stack = deque(pred2_initial_stack, maxlen=FRAME_STACK_SIZE)
                pred2_state = np.concatenate(pred2_frame_stack, axis=2) # State is now a stack of frames
                # Pred 3
                pred3_initial_stack = [pred3_state for _ in range(FRAME_STACK_SIZE)]
                pred3_frame_stack = deque(pred3_initial_stack, maxlen=FRAME_STACK_SIZE)
                pred3_state = np.concatenate(pred3_frame_stack, axis=2) # State is now a stack of frames
            else:
                # Normalise states between 0 and 1
                prey_state = prey_state / max_state
                pred1_state = pred1_state / max_state
                pred2_state =  pred2_state / max_state
                
            
            episode_reward = np.zeros(N_PREDATOR+1)
    
            while True:
                # Get actions
                prey_action = env.action_space.sample()
                pred1_action = pred1.epsilon_greedy_policy(pred1_state, eps, weights1)
                pred2_action = pred2.epsilon_greedy_policy(pred2_state, eps, weights2)
                pred3_action = pred3.epsilon_greedy_policy(pred3_state, eps, weights3)
                actions = [prey_action, pred1_action, pred2_action, pred3_action]
                
                # Take actions, observe next states and rewards
                next_observations, reward_vectors, done, _ = env.step(actions)
                next_prey_state, next_pred1_state, next_pred2_state, next_pred3_state = next_observations
                _, pred1_rewards, pred2_rewards, pred3_rewards = reward_vectors
                #env.render()
                
                # Linear scalarisation
                prey_reward = 0 # Don't care about prey reward for now
                pred1_reward = np.dot(pred1_rewards, pref1)
                pred2_reward = np.dot(pred2_rewards, pref2)
                pred3_reward = np.dot(pred3_rewards, pref3)
                rewards = [prey_reward, pred1_reward, pred2_reward, pred3_reward]
                
                pred1_frame_stack.append(next_pred1_state)
                next_pred1_state = np.concatenate(pred1_frame_stack, axis=2)
    
                pred2_frame_stack.append(next_pred2_state)
                next_pred2_state = np.concatenate(pred2_frame_stack, axis=2)
                
                pred3_frame_stack.append(next_pred3_state)
                next_pred3_state = np.concatenate(pred3_frame_stack, axis=2)

                # Assign next state to current state !!
                pred1_state = next_pred1_state
                pred2_state = next_pred2_state
                pred3_state = next_pred3_state
                
                steps += 1
                episode_reward += np.array(rewards)
            
                
                if done:
                    if pred1_rewards[2]: #or pred2_rewards[2]:
                        lone1_captures += 1
                    elif pred2_rewards[2]:
                        lone2_captures += 1
                    elif pred3_rewards[2]:
                        lone3_captures += 1
                    elif pred1_rewards[3] + pred2_rewards[3] + pred3_rewards[3] == 2:
                        pair_captures += 1
                    elif pred1_rewards[3] + pred2_rewards[3] + pred3_rewards[3] == 3:
                        team_captures += 1
                        
                    print(f'\rEp {episode}: Lone1: {lone1_captures}, Lone2: {lone2_captures}, Lone3: {lone3_captures}, Pair: {pair_captures}, Team: {team_captures}', end='')
                    break
                
        results.append([pref1[2], 
                        pref1[3], 
                        lone1_captures, 
                        lone2_captures, 
                        lone3_captures,
                        lone1_captures + lone2_captures + lone3_captures,
                        pair_captures,
                        team_captures])
                        
    results = pd.DataFrame(results, columns = ["Competitive", 
                                               "Cooperative", 
                                               "Lone 1 Capture", 
                                               "Lone 2 Capture", 
                                               "Lone 3 Capture",
                                               "Lone Capture",
                                               "Pair Capture",
                                               "Team Capture"])
    
    results.to_csv(f'./results/wolfpack_tuning_3predator_DDMMYY.csv')
    run_time = datetime.now() - start_time
    print(f'\nRun time: {run_time} s')
