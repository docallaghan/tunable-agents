# tunable-agents

* Content in ``gym-mo-master`` is from https://github.com/johan-kallstrom/gym-mo with adjustments for using linear scalarisation with all positive real-valued weights for the Gathering environment.
* Content in ``gym-wolfpack-custom`` has been adapted from https://github.com/dkkim93/gym-wolfpack to simplify the grid and convert the environment to a multi-objective one.

## Dependencies
```
Python : 3.6.9 (default, Apr 18 2020, 01:56:04) 
[GCC 8.4.0]

Gym : 0.17.2
NumPy : 1.18.5
TensorFlow : 2.3.0
Matplotlib : 3.2.2
```

## Installation
```
$ git clone https://github.com/docallaghan/tunable-agents.git
$ cd tunable-agents
$ pip install -e gym-mo-master
$ pip install -e gym-wolfpack-custom
```

## File Descriptions
```
deep_sea_treasure/ : All code for Deep Sea Treasure benchmarking experiment (Section 4.1 of thesis)

  deep_sea_treasure.py : Code for linearly scalarised Q-learning in DST environment
  deep_sea_treasure_analysis.py : Code for plotting training progress and computing Pareto front for the trained agent
  models/ : Directory containing pickle file for trained agent in DST environment

---

single_objective_gathering/ : All code for Single-Objective Gathering benchmarking experiment (Section 4.2 of thesis)

  so_gathering.py : Code for training the DQN agent in the single-objective Gathering environment
  so_gathering_plotting.py : Code for plotting the training progress for the trained DQN agent
  models/ : Directory containing the .h5 file for the trained DQN agent
  plots/ : Directory containing the .csv file of the reward data collected during training

---

tunable_gathering/ : All code for the Gathering experiment (Chapter 5 of thesis)

  mo_gathering.py : Code for training tunable and fixed agents using both scalarisation methods in the multi-objective Gathering environment
  mo_gathering_results_tables.py : Code for running simulations with trained agents and generating results tables
  mo_gathering_tuning_performance.py : Code for running simulations with trained agents and generating data for heatmaps
  mo_gathering_heatmaps.Rmd : Code for creating heatmaps showing the agents' tuning performance
  mo_gathering_plotting.py : Code for plotting the training progress for all trained agents
  models/ : Directory containing the .h5 files for all trained agents
  plots/ : Directory containing the .csv files of the reward data collected during training
  results/ : Directory containing results tables and tuning performance data
  
---

tunable_wolfpack/ : All code for the Wolfpack experiment (Chapter 6 of thesis)

  mo_wolfpack.py : Code for training tunable and fixed agents in the multi-objective Wolfpack environment
  mo_wolfpack_tuning_matched.py : Code for running simulations with 2 trained tunable predators with matched objective preferences
  mo_wolfpack_tuning_varied.py : Code for running simulations with 2 trained tunable predators with varied objective preferences
  mo_wolfpack_3pred_sims.py : Code for running simulations with 3 trained tunable predators in the environment
  mo_wolfpack_plotting.py : Code for plotting training progress for all agents and line plots for tuning performance for tunable agents
  mo_wolfpack_heatmaps.Rmd : Code for creating heatmap showing the tunable predators' tunable cooperativeness
  mo_wolfpack_payoff_fixed.py : Code for generating the payoff matrices for the fixed agents
  mo_wolfpack_payoff_tunable.py : Code for generating the payoff matrices for the tunable agents
  models/ : Directory containing the .h5 files for all trained agents
  plots/ : Directory containing the .csv files of the reward data collected during training
  results/ : Directory containing the data collected for visualising tuning performance
```
