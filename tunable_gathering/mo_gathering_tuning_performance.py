# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:55:08 2020

Author: David O'Callaghan
"""

from gym_mo.envs.gridworlds.mo_gathering_env import MOGatheringEnv
from mo_gathering  import DQNAgent

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from collections import deque

    
SEED = 21
EPISODES = 5
FRAME_STACK_SIZE = 3


# Set random seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

run_results = []

env = MOGatheringEnv(penalty_sign=False) # Set to True for scalarisation2
ag = DQNAgent(env)
ag.load_model('./models/dqn_model_tunable_scalarisation1.h5')
agent_vals = [x for x in range(-20, 21) if x % 5 == 0]
red_vals = [x for x in range(-20, 21) if x % 5 == 0]


gathering_tuning_results = {}
for agent_val in agent_vals:
    red_val_results = []
    for red_val in red_vals:
        preference = np.array([-1, -5, +10, red_val, +10, agent_val], dtype=np.float32)
        scal = ag.get_scalarisation_weights(preference)
        #print(np.round(scal, 2))
        
        
        episode_results = []
        for episode in range(EPISODES):
            
            state = ag.env.reset(preference=preference)
            state = np.float32(state) / 255 # Convert to float32 for tf
            weights = ag.normalise(preference[2:])
            
            print(f'\rPref: {preference} , Weights:{weights} , Episode: {episode+1}', end='')
            
            # Create stack
            initial_stack = [state for _ in range(FRAME_STACK_SIZE)]
            ag.frame_stack = deque(initial_stack, maxlen=FRAME_STACK_SIZE)
            state = np.concatenate(ag.frame_stack, axis=2)
            
            steps, wall, green, red, yellow, other_agent_red = [0 for _ in range(6)]
            
            while True:
                action = ag.epsilon_greedy_policy(state, 0.01, weights)
                
                state, rewards, done, _ = ag.env.step(action)
                state = np.float32(state) / 255 # convert to float32 for tf
                
                ag.frame_stack.append(state) # Add to stack
                state = np.concatenate(ag.frame_stack, axis=2)
                
                if rewards[0] != 0:
                    steps += 1
                if rewards[1] != 0:
                    wall += 1
                if rewards[2] != 0:
                    green += 1
                if rewards[3] != 0:
                    red += 1
                if rewards[4] != 0:
                    yellow += 1
                if rewards[5] != 0:
                    other_agent_red += 1
                
                reward = np.dot(scal, rewards)
                    
                if done:
                    episode_results.append([red_val, steps, wall, green, red, yellow, other_agent_red, reward])
                    break
            #print(episode_results)
        red_val_results.append(np.mean(episode_results, axis=0))
        print()
    
    gathering_tuning_results[agent_val] = np.array(red_val_results)
    print()

results = []
for key in gathering_tuning_results:
    for row in gathering_tuning_results[key]:
        results.append([key, *row])
        
results_df = pd.DataFrame(np.array(results), 
                          columns = ["A2R Pref", 
                                     "Red Pref", 
                                     "Steps", 
                                     "Wall", 
                                     "Green", 
                                     "Red", 
                                     "Yellow", 
                                     "A2R", 
                                     "Reward"])

results_df.to_csv('results/tuning_performance_scalarisation1_test.csv')