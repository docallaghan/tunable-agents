# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:42:26 2020

Author: David O'Callaghan
"""

import numpy as np
import random

import tensorflow as tf

from collections import deque # Used for replay buffer and reward tracking

from mo_wolfpack import DQNAgent, env, FRAME_STACK_SIZE

payoff = np.zeros((2,2,2))

PATH_DIR = "./models/"

MODEL_PATHS = [f'{PATH_DIR}/wolfpack_model_tunable_pred1_seed1.h5',
               f'{PATH_DIR}/wolfpack_model_tunable_pred2_seed1.h5']

def play_game(s1, s2):
    # Decay epsilon
    eps = 0.01
    # Reset env
    observations = env.reset()

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
    
    episode_reward = np.zeros(2)
    steps = 0
    while True:
        
        # Get actions
        prey_action = env.action_space.sample()
        pred1_action = pred1.epsilon_greedy_policy(pred1_state, eps, s1[2:])
        pred2_action = pred2.epsilon_greedy_policy(pred2_state, eps, s2[2:])
        actions = [prey_action, pred1_action, pred2_action]

        # Take actions, observe next states and rewards
        next_observations, reward_vectors, done, _ = env.step(actions)
        next_prey_state, next_pred1_state, next_pred2_state = next_observations
        _, pred1_rewards, pred2_rewards = reward_vectors
        
        pred1_reward = np.sum(pred1_rewards[2:])
        pred2_reward = np.sum(pred2_rewards[2:])
        rewards = [pred1_reward, pred2_reward]
        
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
        # env.render()
        steps += 1
        episode_reward += np.array(rewards)
        
        if done:
            print(s1, s2, episode_reward)
            break
                
    return episode_reward
        

if __name__ == '__main__':
    
    random.seed(42)
    np.random.seed(42)
    env.seed(42)
    tf.random.set_seed(42)
    EPISODES = 250
    
    pred1 = DQNAgent(1) ; pred1.load_model(MODEL_PATHS[0])
    pred2 = DQNAgent(2) ; pred2.load_model(MODEL_PATHS[1])
    
    cooper = np.array([0.005, 0.025, 0, 0.97], dtype=np.float32)    
    defect = np.array([0.005, 0.025, 0.97, 0], dtype=np.float32)
    
    strats = [cooper, defect]
    
    # 0 - Cooperatate, 1 - Defect
    strat_ids = [[0,0],
                 [0,1],
                 [1,0],
                 [1,1]]
    
    for s1, s2 in strat_ids:
        for i in range(EPISODES):
            print(i+1, end='  ')
            r1, r2 = play_game(strats[s1], strats[s2])
            payoff[0,s1,s2] += r1 / EPISODES
            payoff[1,s1,s2] += r2 / EPISODES
    
    print('\n\n')
    print(payoff)
