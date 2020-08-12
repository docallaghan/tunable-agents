# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:38:16 2020

Author: David O'Callaghan
"""

from gym_mo.envs.gridworlds.mo_gathering_env import MOGatheringEnv
from mo_gathering  import DQNAgent

import numpy as np
import pandas as pd
import random
import tensorflow as tf

from datetime import datetime # Used for timing script
from collections import deque
import os


SEED = 21
EPISODES = 250
FRAME_STACK_SIZE = 3

model_dir = './models/'
out_dir = './results/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Scalarisation method 1
PENALTY_SIGN = False
fixed_agent_paths = [
    model_dir + 'dqn_model_fixed1_scalarisation1.h5',
    model_dir + 'dqn_model_fixed2_scalarisation1.h5',
    model_dir + 'dqn_model_fixed3_scalarisation1.h5',
    model_dir + 'dqn_model_fixed4_scalarisation1.h5',
    ]
tunable_agent_path = model_dir + 'dqn_model_tunable_scalarisation1.h5'

# # Scalarisation method 2
# PENALTY_SIGN = True
# fixed_agent_paths = [
#     model_dir + 'dqn_model_fixed1_scalarisation2.h5',
#     model_dir + 'dqn_model_fixed2_scalarisation2.h5',
#     model_dir + 'dqn_model_fixed3_scalarisation2.h5',
#     model_dir + 'dqn_model_fixed4_scalarisation2.h5',
#     ]
# tunable_agent_path = model_dir + 'dqn_model_tunable_scalarisation2.h5'

fixed_preferences_dict = {
    '1': np.array([-1, -5, +10, +20, +10, -20]), # Competitive
    '2': np.array([-1, -5, +10, +20, +10, +20]), # Cooperative
    '3': np.array([-1, -5, +20, +15, +20, +20]), # Fair
    '4': np.array([-1, -5, +20,   0, +20, +20]), # Generous
    }

base_dict = {'steps':0, 'green':0, 'red':0, 'yellow':0, 'other-agent-red':0}

def play_episode(agent, preference, results_dict):
    """
    Play one episode using the DQN and display the grid image at each step.
    """
    state = agent.env.reset(preference=preference)
    state = np.float32(state) / 255 # Convert to float32 for tf
    weights = agent.normalise(preference[2:])
    # weights = agent.normalise(preference[[3,5]])
    #print(preference, weights)
    
    # Create stack
    initial_stack = [state for _ in range(FRAME_STACK_SIZE)]
    agent.frame_stack = deque(initial_stack, maxlen=FRAME_STACK_SIZE)
    state = np.concatenate(agent.frame_stack, axis=2)

    while True:
        # Take eps greedy action
        action = agent.epsilon_greedy_policy(state, 0.01, weights)
        state, rewards, done, _ = agent.env.step(action)
        state = np.float32(state) / 255 # convert to float32 for tf
        
        # Add state to frame stack
        agent.frame_stack.append(state)
        state = np.concatenate(agent.frame_stack, axis=2)
        
        # Record results
        results_dict['steps'] += 1
        if abs(rewards[2]) == 1:
            results_dict['green'] += 1
        if abs(rewards[3]) == 1:
            results_dict['red'] += 1
        if abs(rewards[4]) == 1:
            results_dict['yellow'] += 1
        if abs(rewards[5]) == 1:
            results_dict['other-agent-red'] += 1
        
        if done:
            break
        
def summarise_and_save(df_in, func, path):
    df_out = pd.DataFrame()
    for pref_id in tunable_results_by_pref:
        df_out[pref_id] = df_in[pref_id].apply(lambda x: func(x))
    df_out = df_out.transpose()
    df_out.index.name = 'Behaviour'
    df_out.to_csv(path)
    return df_out


if __name__ == '__main__':
    # Set random seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    
    # For timing the script
    start_time = datetime.now()
    
    # Instantiate environment
    item_env1 = MOGatheringEnv(from_pixels=True, penalty_sign=PENALTY_SIGN)
    item_env2 = MOGatheringEnv(from_pixels=True, penalty_sign=PENALTY_SIGN)
    
    # Instantiate tunable agent, pass in env and load model
    tunable_agent = DQNAgent(item_env1)
    tunable_agent.load_model(tunable_agent_path)
    
    # For storing results for each preference
    tunable_results_by_pref = {}
    fixed_results_by_pref = {}
    
    # Iterate through models
    for pref_id, fixed_agent_path in zip(fixed_preferences_dict, fixed_agent_paths):
        # Instantiate fixed agent, pass in env and load model
        fixed_agent = DQNAgent(item_env2)
        fixed_agent.load_model(fixed_agent_path)
        
        tunable_results = []
        fixed_results = []
                
        for episode in range(EPISODES):
            tunable_dict = base_dict.copy()
            fixed_dict = base_dict.copy()
            
            # Play epsiode with tunable agent
            play_episode(agent=tunable_agent, 
                          preference=fixed_preferences_dict[pref_id],
                          results_dict=tunable_dict)
            
            # Play episode with fixed agent
            play_episode(agent=fixed_agent, 
                          preference=fixed_preferences_dict[pref_id],
                          results_dict=fixed_dict)
            
            # Store collected results from episode
            tunable_results.append(tunable_dict.copy())
            fixed_results.append(fixed_dict.copy())
            
            print(f'\rEpisode: {episode+1}  Preference: {pref_id}', end="")
        
        # Store collected results for preference across all episodes
        tunable_results_by_pref[pref_id] = pd.DataFrame(tunable_results)
        fixed_results_by_pref[pref_id] = pd.DataFrame(fixed_results)
        
    # Compute mean and sd across episodes and save to csv
    path_id = 'DDMMYY'
    tunable_mean = summarise_and_save(tunable_results_by_pref, np.mean, f'{out_dir}/tunable_mean_{path_id}.csv')
    tunable_std = summarise_and_save(tunable_results_by_pref, np.std, f'{out_dir}/tunable_std_{path_id}.csv')
    fixed_mean = summarise_and_save(fixed_results_by_pref, np.mean, f'{out_dir}/fixed_mean_{path_id}.csv')
    fixed_std = summarise_and_save(fixed_results_by_pref, np.std, f'{out_dir}/fixed_std_{path_id}.csv')
     
    run_time = datetime.now() - start_time
    print(f'Run time: {run_time} s')
    