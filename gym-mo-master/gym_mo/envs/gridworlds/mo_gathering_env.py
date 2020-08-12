import time

from gym_mo.envs.gridworlds import gridworld_base
from gym_mo.envs.gridworlds.mo_gridworld_base import MOGridworld
from gym_mo.envs.gridworlds.gridworld_base import GridObject, HunterAgent

import numpy as np
import random

GATHERING_MAPPING = { # is_walkable, is_consumed, reward_on_encounter, color, idx
    '#': GridObject(True, False, 0, (255.0, 255.0, 255.0), 1), # White (Not used)
    'o': GridObject(True, True, 0, (0.0, 255.0, 0.0), 2), # Green
    'p': GridObject(True, True, 1, (255.0, 0.0, 0.0), 3), # Red
    'q': GridObject(True, True, 0, (255.0, 255.0, 0.0), 4), # Yellow
    ' ': None # Empty cell
}

GATHERING_MAP = [
'        ',
'        ',
'        ',
'        ',
'        ',
'        ',
'        ',
'        ',
]


class MOGatheringEnv(MOGridworld):

    def __init__(self,
                 from_pixels=True,
                 agent_start=[0,0], # Bottom left
                 agent_color=(0.0, 0.0, 255.0), # Blue
                 preference=np.array([-1,-5,+20,-20,-20,+0]),
                 random_items=['p','o','p','o','p','o','q','q'],
                 random_items_frame=2, # Leave frame of 2 spaces around random items
                 penalty_sign=False,
                 agents=[]):

        agent0 = HunterAgent(3, True, False, 0, (255.0, 0.0, 255.0), 5) # Pink
        agent0.set_position([7,7]) # Top right

        GATHERING_AGENTS = [agent0]

        super(MOGatheringEnv, self).__init__(map=GATHERING_MAP,
                                             object_mapping=GATHERING_MAPPING,
                                             random_items=random_items,
                                             random_items_frame=random_items_frame,
                                             from_pixels=from_pixels,
                                             init_agents=GATHERING_AGENTS,
                                             agent_start=agent_start,
                                             agent_color=agent_color,
                                             preference=preference,
                                             penalty_sign=penalty_sign,
                                             max_steps=30, include_agents=False)


if __name__=="__main__":
    my_grid = MOGatheringEnv(from_pixels=True, 
                             penalty_sign=True, 
                             preference=np.array([-1,-5,0,0,5,20]))

    
    for _ in range(20):
        done = False
        my_grid.reset()
        while not done:
            _, r, done, _ = my_grid.step(random.choice([1,2,3,4]))
            if np.any(r[2:]!=0):    
                print(r)
            #my_grid.render()
            time.sleep(0.01)
        print()

