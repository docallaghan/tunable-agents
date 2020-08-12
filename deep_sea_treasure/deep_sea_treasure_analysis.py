# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:50:22 2020

Author: David O'Callaghan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import style
from deep_sea_treasure import DeepSeaTreasureEnvironment

style.use('ggplot')
colour_palette = get_cmap(name='tab10').colors

def is_pareto_efficient(costs):
    """
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    
    Find the pareto-efficient points
    
    :param costs: An (n_points, n_costs) array
    
    :return: is_efficient (n_points, ) boolean array, indicating whether each 
    point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

if __name__ == '__main__':
    with open('models/dst_results.pkl', 'rb') as f:
        stats_dict = pickle.load(f)
    
    data = []
    for key in stats_dict:
        Q_values = stats_dict[key][1]
        env = DeepSeaTreasureEnvironment()
        state = env.reset()
        r = [0,0]
        while True:
            action = np.argmax(Q_values[(*state,)])
            state, rewards, done = env.step(action)
            r[0] += rewards[0]
            r[1] += rewards[1]
            if done:
                print(f'Finished {key}')
                print(f'{r}\n')
                data.append(r)
                break
            
    data = np.array(data)
    # Filter the pareto front
    est_pareto = data[is_pareto_efficient(data)]
    true_pareto = np.array([[-1,-3,-5,-7,-8,-9,-13,-14,-17,-19],
                            [1,34,58,78,86,92,112,116,122,124]]).T
    # Plot
    fig1, ax = plt.subplots()
    ax.scatter(true_pareto[:,0], true_pareto[:,1], s=80, c=colour_palette[0], marker='x', label='True Pareto Front')
    ax.scatter(est_pareto[:,0], est_pareto[:,1], s=80, c=colour_palette[1], marker='+', label='Found Pareto Front')
    ax.legend(facecolor='white')
    ax.set_xlabel('Time Penalty')
    ax.set_ylabel('Treasure Value')
    #ax.set_title('Deep Sea Treasure')
    ax.set_facecolor('white')
    ax.legend(facecolor='white')
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    ax.grid(True, ls=':', c='dimgrey')
    # # plt.show()

    
    ps = np.arange(0,1,0.05)
    row, col = 0, 0
    nrows, ncols = 4, 5
    fig2, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(18,9))
    # Add a frame to give common axis labels
    fig2.add_subplot(111, frameon=False)
    for p in ps:
        print(row, col)
        key = tuple([round(p, 2), round(1-p, 2)])
        axs[row, col].plot(stats_dict[key][0][:,0], 
                            stats_dict[key][0][:,1],
                            label=key,
                            c=colour_palette[0],
                            alpha=0.9, linewidth=0.7)
        axs[row, col].set_facecolor('white')
        axs[row, col].legend(facecolor='white', loc='lower left')
        axs[row, col].xaxis.set_ticks_position('none') 
        axs[row, col].yaxis.set_ticks_position('none') 
        axs[row, col].grid(True, ls=':', c='dimgrey')
        if col == ncols - 1:
            col = 0
            row += 1
        else:
            col += 1
    
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.show()
