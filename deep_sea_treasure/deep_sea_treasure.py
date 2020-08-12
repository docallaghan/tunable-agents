# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:53:15 2020
Author: David O'Callaghan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import random


SEED = 42


class DeepSeaTreasureEnvironment:
    
    grid_rows = 11
    grid_cols = 10
    
    depths = [1, 2, 3, 4, 4, 4, 7, 7, 9, 10]
    #treasure = [0.5, 28, 52, 73, 82, 90, 115, 120, 134, 143]
    treasure = [1, 34, 58, 78, 86, 92, 112, 116, 122, 124]
    
    # UP, DOWN, LEFT, RIGHT
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    
    def __init__(self):
        self.reset()
        self.forbidden_states = self.__get_forbidden_states()
        self.treasure_locations = self.__get_treasure_locations()
    
    def __get_forbidden_states(self):
        forbidden_states = [(i, j) for j in range(self.grid_cols) 
                            for i in range(self.depths[j]+1, self.grid_rows)]
        return forbidden_states
    
    def __get_treasure_locations(self):
        treasure_locations = [(i, j) for j, i in enumerate(self.depths)]
        return treasure_locations
            
    def reset(self):
        self.n_steps = 0
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """
        Transition the environment through the input action
        """
        self.n_steps += 1
        # "Candidate" next location for the agent
        cand_loc = (self.state[0] + self.actions[action][0], 
                    self.state[1] + self.actions[action][1])
    

        # Check if forbidden state
        if ((cand_loc[0] <= self.grid_rows-1 and cand_loc[0] >= 0) and
            (cand_loc[1] <= self.grid_cols-1 and cand_loc[1] >= 0) and
            (cand_loc not in self.forbidden_states)):
            # Set new state
            self.state = cand_loc
        
        rewards = self.get_rewards()
        state = self.state
        done = self.check_terminal_state()
        return state, rewards, done
    
    def get_rewards(self):
        rewards = [-1, 0] # (time_penalty, treasure_reward)
        if self.state in self.treasure_locations:
            rewards[1] = self.treasure[self.state[1]]
        return tuple(rewards)
    
    def check_terminal_state(self):
        return (self.state in self.treasure_locations) or (self.n_steps > 200)


class Agent:
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(len(env.actions))]
    
    def epsillon_greedy_policy(self, state, epsillon):
        if np.random.rand() < epsillon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.Q_values[state])
        
    def scalarise(self, rewards, weights):
        
        rewards = np.array(rewards)
        return np.dot(rewards, weights)
    
    def initialise_q_values(self):
        # Q_values = np.random.randint(0, 125, size=(self.env.grid_rows, 
        #                                            self.env.grid_cols, 
        #                                            len(self.env.actions)))
        Q_values = np.random.rand(self.env.grid_rows,
                                  self.env.grid_cols,
                                  len(self.env.actions)) * 100
        
        for forbidden_state in self.env.forbidden_states:
            Q_values[forbidden_state] = np.full(len(self.env.actions), -200)
            
        for treasure_location in self.env.treasure_locations:
            Q_values[treasure_location] = np.zeros(len(self.env.actions))
            
        print(np.round(np.max(Q_values, axis=2),1))
        return Q_values
    
    @staticmethod
    def weights_gen(n):
        w0 = 0
        while w0 <= 1.0:
            w1 = 1.0 - w0
            yield np.array([w0, w1])
            w0 += 1 / (n-1)
            
    def q_learning(self, episodes):

        #alpha0 = 0.1 # initial learning rate
        epsillon0 = 0.998
        alpha = 0.1
        gamma = 1

        self.stats_dict = {}
        for weights in self.weights_gen(101):
            self.Q_values = self.initialise_q_values()
            stats = []
            for i in range(episodes):
                state = self.env.reset()
                rs = 0
    
                #alpha = max(alpha0 - i / episodes, 0.001) # decay learning rate
                #epsillon = max(epsillon0 - i / episodes, 0.05) # decay epsilon
                epsillon = epsillon0 ** i
    
                while True:
                    action = self.epsillon_greedy_policy(state, epsillon)
                    #print(action)
                    next_state, rewards, done = self.env.step(action)
                    reward = self.scalarise(rewards, weights)
                    rs += reward
                    self.Q_values[(*state, action)] += alpha * (reward +  gamma *
                                                np.max(self.Q_values[next_state]) - 
                                                self.Q_values[(*state, action)])
                    if done:
                        break
                    state = next_state
                stats.append([i, rs])
            key = tuple(np.round(weights, 4))
            self.stats_dict[key] = [np.array(stats), self.Q_values.copy()]
            #self.plot_learning_curve(self.stats_dict[key][0], key)
        
        with open('models/dst_results.pkl', 'wb') as f:
            pickle.dump(self.stats_dict, f)
        
    def plot_learning_curve(self, stats, key):
        """
        Plot the rewards per episode collected during training
        """
        fig, ax = plt.subplots()
        ax.plot(stats[:,0], stats[:,1])
        ax.set_xlabel('episode')
        ax.set_ylabel('reward per episode')
        ax.set_title(f'time, treasure weighting: {key}')
        plt.show()


if __name__ == '__main__':
    np.random.seed(SEED)
    random.seed(SEED)
    
    dst_env = DeepSeaTreasureEnvironment()
    ag = Agent(dst_env)
    ag.q_learning(4000)

    fig, ax = plt.subplots()
    ax.plot(ag.stats_dict[(0.5, 0.5)][0][:,0], 
            ag.stats_dict[(0.5, 0.5)][0][:,1])
