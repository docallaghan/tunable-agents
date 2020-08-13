# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:07:50 2020

Author: David O'Callaghan
"""

import numpy as np
import random
import pandas as pd

import tensorflow as tf

from collections import deque # Used for replay buffer and reward tracking

from mo_wolfpack import DQNAgent, env, FRAME_STACK_SIZE, N_PREDATOR

PATH_DIR = "./models/"

MODEL_PATHS = [f'{PATH_DIR}/wolfpack_model_tunable_pred1_seed1.h5',
               f'{PATH_DIR}/wolfpack_model_tunable_pred2_seed1.h5']

EPISODES = 250

class PreferenceSpace:
    
    def __init__(self):
        w0 = 0.005 # Time penalty
        w1 = 5 * w0 # Wall penalty : 5x time penalty
        w2_range = np.linspace(0,0.97,5)
        self.distribution = [np.array([w0, w1, w2, 0.97 - w2], dtype=np.float32) for w2 in w2_range]
        
    def sample(self):
        return random.choice(self.distribution)

if __name__ == '__main__':
    
    random.seed(42)
    np.random.seed(42)
    env.seed(42)
    tf.random.set_seed(42)

    ps = PreferenceSpace()
    
    # Initialise agents
    # prey1 = DQNAgent(0)
    pred1 = DQNAgent(1)
    pred2 = DQNAgent(2)   
    
    pred1.load_model(MODEL_PATHS[0])
    pred2.load_model(MODEL_PATHS[1])
    
    steps = 0
    
    prefs = np.linspace(0, 0.97, 17)
    results = []
    for pref in prefs:
        
        steps = 0
    
        lone1_captures = 0
        lone2_captures = 0
        team_captures = 0
        
        pref1 = np.array([0.005, 0.025, pref, 0.97 - pref], dtype=np.float32)
        pref2 = pref1.copy()
        
        weights1 = pref1[2:]
        weights2 = pref2[2:]
        
        print(f'\n\n{np.round(pref1[2:], 3)}\n-----------------')
        
        for episode in range(1, EPISODES+1):
            # Decay epsilon
            eps = 0.01
            # Reset env
            observations = env.reset()
            #env.render()
            prey_state, pred1_state, pred2_state = observations
               
            # Create deque for storing stack of N frames
            # Pred 1
            pred1_initial_stack = [pred1_state for _ in range(FRAME_STACK_SIZE)]
            pred1_frame_stack = deque(pred1_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred1_state = np.concatenate(pred1_frame_stack, axis=2) # State is now a stack of frames
            # Pred 2
            pred2_initial_stack = [pred2_state for _ in range(FRAME_STACK_SIZE)]
            pred2_frame_stack = deque(pred2_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred2_state = np.concatenate(pred2_frame_stack, axis=2) # State is now a stack of frames
            
            episode_reward = np.zeros(N_PREDATOR+1)
    
            while True:
                
                # Get actions
                prey_action = env.action_space.sample()
                pred1_action = pred1.epsilon_greedy_policy(pred1_state, eps, weights1)
                pred2_action = pred2.epsilon_greedy_policy(pred2_state, eps, weights2)
                actions = [prey_action, pred1_action, pred2_action]
                
                
                # Take actions, observe next states and rewards
                next_observations, reward_vectors, done, _ = env.step(actions)
                next_prey_state, next_pred1_state, next_pred2_state = next_observations
                _, pred1_rewards, pred2_rewards = reward_vectors
                
                # Linear scalarisation
                prey_reward = 0 # Don't care about prey reward for now
                pred1_reward = np.dot(pred1_rewards, pref1)
                pred2_reward = np.dot(pred2_rewards, pref2)
                rewards = [prey_reward, pred1_reward, pred2_reward]
                
                # Store in replay buffers
                # Pred 1
                pred1_frame_stack.append(next_pred1_state)
                next_pred1_state = np.concatenate(pred1_frame_stack, axis=2)
                
                # Pred 2
                pred2_frame_stack.append(next_pred2_state)
                next_pred2_state = np.concatenate(pred2_frame_stack, axis=2)
                
                # Assign next state to current state !!
                pred1_state = next_pred1_state
                pred2_state = next_pred2_state
                
                steps += 1
                episode_reward += np.array(rewards)
                
                if done:
                    if pred1_rewards[2]: #or pred2_rewards[2]:
                        lone1_captures += 1
                    elif pred2_rewards[2]:
                        lone2_captures += 1
                    elif pred1_rewards[3] and pred2_rewards[3]:
                        team_captures += 1
                        
                    print(f'\rEp {episode}: Lone1: {lone1_captures}, Lone2: {lone2_captures}, Team: {team_captures}', end='')
                    break
            
        results.append([pref1[2], pref1[3], lone1_captures, lone2_captures, team_captures])

            
    results = pd.DataFrame(results, columns = ["Competitive", "Cooperative", "Lone 1 Capture", "Lone 2 Capture", "Team Capture"])
    
    results.to_csv(f'./results/wolfpack_tuning_matched_prefs_DDMMYY.csv')
