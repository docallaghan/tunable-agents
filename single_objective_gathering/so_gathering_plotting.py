# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:22:41 2020

Author: David O'Callaghan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import style
from collections import deque

style.use('ggplot')

colour_palette = get_cmap(name='tab10').colors

data_dir = './plots/'

MEAN_EVERY = 100

fig, ax = plt.subplots(nrows=1, ncols=1)


class MovingAverage(deque):
    def mean(self):
        return sum(self) / len(self)
    
    
def plot_reward_data(csv_path, plot_id, colour_id, legend_label):
    reward_data = np.loadtxt(csv_path, delimiter=',')
    mean_rewards = np.zeros(len(reward_data))
    tracker = MovingAverage(maxlen=MEAN_EVERY)

    for j, (_, reward) in enumerate(reward_data):
        tracker.append(reward)
        mean_rewards[j] = tracker.mean()
    
    ax.plot(reward_data[:,0], reward_data[:,1], 
        c=colour_palette[colour_id], alpha=0.3)
    ax.plot(reward_data[MEAN_EVERY//2:,0], mean_rewards[MEAN_EVERY//2:], 
        c=colour_palette[colour_id], label=legend_label, alpha=1)
    print('Plot finished')
    

if __name__ == '__main__':
    plot_reward_data(f'{data_dir}/reward_data_single_objective_gathering.csv', 0, 0, 'DQN Agent')

    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Mean {MEAN_EVERY} Episode Reward')
    #ax.set_title('Training Progress for Item Gathering Environment')
    ax.grid(True, ls=':', c='dimgrey')
    ax.set_facecolor('white')
    ax.legend(facecolor='white')

    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    plt.show()
