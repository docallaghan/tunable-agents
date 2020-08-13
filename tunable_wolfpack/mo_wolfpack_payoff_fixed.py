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


def play_game(ag1, ag2):
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
    
    episode_reward = np.zeros(2)
    steps = 0
    while True:
        
        # Get actions
        prey_action = env.action_space.sample()
        pred1_action = ag1.epsilon_greedy_policy(pred1_state, eps, ag1.pref[2:])
        pred2_action = ag2.epsilon_greedy_policy(pred2_state, eps, ag2.pref[2:])
        actions = [prey_action, pred1_action, pred2_action]
        
        # Take actions, observe next states and rewards
        next_observations, reward_vectors, done, _ = env.step(actions)
        next_prey_state, next_pred1_state, next_pred2_state = next_observations
        _, pred1_rewards, pred2_rewards = reward_vectors
        
        # Linear scalarisation
        pred1_reward = np.dot(pred1_rewards[2:], ag1.pref[2:]) / 0.97
        pred2_reward = np.dot(pred2_rewards[2:], ag2.pref[2:]) / 0.97
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
            #print(s1, s2, episode_reward)
            break
                
    return episode_reward
        

if __name__ == '__main__':
    
    random.seed(42)
    np.random.seed(42)
    env.seed(42)
    tf.random.set_seed(42)
    
    EPISODES = 250
    
    cooper = np.array([0.005, 0.025, 0, 0.97], dtype=np.float32)    
    defect = np.array([0.005, 0.025, 0.97, 0], dtype=np.float32)

    PATH_DIR = "./models/"
    
    # Pred 1 and Pred 2 are the same models
    # C1 = DQNAgent(1) ; C1.load_model(f'{PATH_DIR}/wolfpack_model_fixed_cooperative_pred1.h5')
    # C2 = DQNAgent(2) ; C2.load_model(f'{PATH_DIR}/wolfpack_model_fixed_cooperative_pred1.h5')
    # D1 = DQNAgent(3) ; D1.load_model(f'{PATH_DIR}/wolfpack_model_fixed_competitive_pred1.h5')
    # D2 = DQNAgent(4) ; D2.load_model(f'{PATH_DIR}/wolfpack_model_fixed_competitive_pred1.h5')
    
    # Pred 2 models are trained with different seed
    C1 = DQNAgent(1) ; C1.load_model(f'{PATH_DIR}/wolfpack_model_fixed_cooperative_pred1.h5')
    C2 = DQNAgent(2) ; C2.load_model(f'{PATH_DIR}/wolfpack_model_fixed_cooperative_pred2.h5')
    D1 = DQNAgent(3) ; D1.load_model(f'{PATH_DIR}/wolfpack_model_fixed_competitive_pred1.h5')
    D2 = DQNAgent(4) ; D2.load_model(f'{PATH_DIR}/wolfpack_model_fixed_competitive_pred2.h5')
    
    C1.pref = cooper.copy()
    C2.pref = cooper.copy()
    D1.pref = defect.copy()
    D2.pref = defect.copy()
    
    agents = [C1, C2, D1, D2]
    
    games = [[0,1],# C v C
             [0,3],# C v D
             [2,1],# D v C
             [2,3] # D v D
            ]
    
    # 0 - Cooperatate, 1 - Defect
    strat_ids = [[0,0],
                 [0,1],
                 [1,0],
                 [1,1]]
    
    for (ag1, ag2), (s1, s2) in zip(games, strat_ids):
        for i in range(EPISODES):
            print(f'\r(s1,s2)=({s1},{s2})  Ep = {i}', end='')

            r1, r2 = play_game(agents[ag1], agents[ag2])

            payoff[0,s1,s2] += r1 / EPISODES
            payoff[1,s1,s2] += r2 / EPISODES

        print()
    
    print('\n\n')
    print(payoff)