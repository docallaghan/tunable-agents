# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:19:28 2020

Author: David O'Callaghan
"""


import numpy as np
import random
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from collections import deque
from datetime import datetime

from gym_wolfpack_custom.envs import MOWolfpackCustomEnv


SEED = 42
IMAGE = True

FRAME_STACK_SIZE = 3


class DQNAgent:
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.actions = [i for i in range(env.action_space.n)] 
        
        if IMAGE:
            image_size = env.observation_space.shape
            self.input_size = (*image_size[:2],image_size[-1]*FRAME_STACK_SIZE)
        else:
            self.input_size = env.observation_space.shape

        self.output_size = env.action_space.n
    
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
    
    def load_model(self, path):
        self.model = keras.models.load_model(path)
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
    

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
    
    # 9 evenly spaced preference weights
    prefs = np.linspace(0, 0.97, 9)
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
                    if pred1_rewards[2]:
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
