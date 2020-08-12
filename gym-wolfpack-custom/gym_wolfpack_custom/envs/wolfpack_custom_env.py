import os
import numpy as np
import matplotlib.pyplot as plt

import gym
# from gym import error, spaces, utils
# from gym.utils import seeding


class WolfpackCustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    REWARD_LONELY = 2.
    REWARD_TEAM = 2.
    REWARD_STEP = -0.01
    CAPTURE_RADIUS = 2.
    MAX_STEPS = 250
    
    def __init__(self, n_predator=2):
        super(WolfpackCustomEnv, self).__init__()
        
        self.n_predator = n_predator
        self.config = Config()
        
        self.base_gridmap_array = self._load_gridmap_array()
        self.base_gridmap_image = self._to_image(self.base_gridmap_array)
        
        # self.observation_shape = (11, 11, 3)  # Format: (height, width, channel)
        self.observation_shape = self.base_gridmap_image.shape
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.observation_shape)
        self.action_space = gym.spaces.Discrete(len(self.config.action_dict))
        # self.pad = np.max(self.observation_shape) - 2
    
    def _load_gridmap_array(self):
        """
        Loads the maze text file into a numpy array based on spaces and line
        breaks.
        """
        # Ref: https://github.com/xinleipan/gym-gridworld/blob/master/gym_gridworld/envs/gridworld_env.py
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "maze_small.txt")
        with open(path, 'r') as f:
            gridmap = f.readlines()

        gridmap_array = np.array(
            list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), gridmap)))
        return gridmap_array
    
    def _to_image(self, gridmap_array):
        """
        Converts the array of the grid into an RGB image based on the object
        identifiers. e.g, 0 -> empty , 1 -> wall , ...
        """
        image = np.zeros((gridmap_array.shape[0], gridmap_array.shape[1], 3), dtype=np.float32)

        for row in range(gridmap_array.shape[0]):
            for col in range(gridmap_array.shape[1]):
                grid = gridmap_array[row, col]

                if grid == self.config.grid_dict["empty"]:
                    image[row, col] = self.config.color_dict["empty"]
                elif grid == self.config.grid_dict["wall"]:
                    image[row, col] = self.config.color_dict["wall"]
                elif grid == self.config.grid_dict["prey"]:
                    image[row, col] = self.config.color_dict["prey"]
                elif grid == self.config.grid_dict["predator"]:
                    image[row, col] = self.config.color_dict["predator"]
                # elif grid == self.config.grid_dict["orientation"]:
                #     image[row, col] = self.config.color_dict["orientation"]
                else:
                    raise ValueError()

        return image
    
    def _reset_agents(self):
        self.agents = []
        for i_agent, agent_type in enumerate(["prey"] + ["predator" for _ in range(self.n_predator)]):
            agent = Agent(i_agent, agent_type, self.base_gridmap_array)
            self.agents.append(agent)
            
    def _render_gridmap(self):
        """
        Colours the base grid image at the current locations of the agents
        and their orientations.
        """
        gridmap_image = np.copy(self.base_gridmap_image)

        # # Render orientation
        # for agent in self.agents:
        #     orientation_location = agent.orientation_location
        #     gridmap_image[orientation_location[0], orientation_location[1]] = self.config.color_dict["orientation"]
            
        # Render location
        for agent in self.agents:
            location = agent.location
            gridmap_image[location[0], location[1]] = self.config.color_dict[agent.type]

        # Pad image
        # pad_width = ((self.pad, self.pad), (self.pad, self.pad), (0, 0))
        # gridmap_image = np.pad(gridmap_image, pad_width, mode="constant")

        return gridmap_image
    
    def step(self, actions):
        assert len(actions) == self.n_predator + 1

        # Compute next locations
        for agent, action in zip(self.agents, actions):
            action = list(self.config.action_dict.keys())[action]

            if "spin" not in action: 
                next_location = agent.location + self.config.action_dict[action]
                # next_orientation = agent.orientation
            else:
                next_location = agent.location
                # next_orientation = agent.orientation + self.config.action_dict[action]
            agent.location = next_location
            # agent.orientation = next_orientation

        # Get next observations
        gridmap_image = self._render_gridmap()

        observations = []
        for agent in self.agents:
            observation = self._get_observation(agent, gridmap_image)
            observations.append(observation)

        # Find who succeeded in hunting
        hunted_predator = None
        for predator in self.agents[1:]:
            if np.array_equal(self.agents[0].location, predator.location):
                hunted_predator = predator

        # Find nearby predators to the one succeeded in hunting
        nearby_predators = []
        if hunted_predator is not None:
            for predator in self.agents[1:]:
                if predator.id != hunted_predator.id:
                    dist = np.linalg.norm(predator.location - hunted_predator.location)
                    if dist < self.CAPTURE_RADIUS:
                        nearby_predators.append(predator)

        # Compute reward
        rewards = [self.REWARD_STEP for _ in range(len(self.agents))]
        if hunted_predator is not None:
            if len(nearby_predators) == 0:
                rewards[hunted_predator.id] = self.REWARD_LONELY
            else:
                rewards[hunted_predator.id] = self.REWARD_TEAM
                for neaby_predator in nearby_predators:
                    rewards[neaby_predator.id] = self.REWARD_TEAM
        
        self.n_steps += 1
        
        # Compute done
        if (hunted_predator is not None) or (self.n_steps >= self.MAX_STEPS):
            done = True
        else:
            done = False

        return observations, rewards, done, {}
    
    def reset(self):
        self.n_steps = 0
        self._reset_agents()
        gridmap_image = self._render_gridmap()

        observations = []
        for agent in self.agents:
            observation = self._get_observation(agent, gridmap_image)
            observations.append(observation)

        return observations
    
    def _get_observation(self, agent, gridmap_image):
        """As in  Leibo et al., AAMAS-17 (https://arxiv.org/pdf/1702.03037.pdf),
        the observation depends on each player’s current position and orientation. 
        Specifically, depending on the orientation, the image is cropped and then
        post-processed such that the player's location is always at the bottom center.
        """
        # row, col = agent.location[0] + self.pad, agent.location[1] + self.pad
        # height, half_width = self.observation_shape[0], int(self.observation_shape[1] / 2)

        # if agent.orientation == self.config.orientation_dict["up"]:
        #     observation = gridmap_image[
        #         row - height + 1: row + 1, 
        #         col - half_width: col + half_width + 1, :]
        # elif agent.orientation == self.config.orientation_dict["right"]:
        #     observation = gridmap_image[
        #         row - half_width: row + half_width + 1, 
        #         col: col + height, :]
        #     observation = np.rot90(observation, k=1)
        # elif agent.orientation == self.config.orientation_dict["down"]:
        #     observation = gridmap_image[
        #         row: row + height, 
        #         col - half_width: col + half_width + 1, :]
        #     observation = np.rot90(observation, k=2)
        # elif agent.orientation == self.config.orientation_dict["left"]:
        #     observation = gridmap_image[
        #         row - half_width: row + half_width + 1, 
        #         col - height + 1: col + 1, :]
        #     observation = np.rot90(observation, k=3)
        # else:
        #     raise ValueError()

        # assert observation.shape == self.observation_shape

        # return observation
        observation = gridmap_image.copy()
        if agent.type == 'prey':
            # Prey sees predators in blue
            return observation
        else:
            # Predators see prey in red and other predators in green
            for agent_x in self.agents:
                if agent_x.id != agent.id and agent_x.type != 'prey':
                    location = agent_x.location
                    observation[location[0], location[1]] = self.config.color_dict['other_predator']
            return observation
    
    def render(self, mode='human', close=False):
        gridmap_image = self._render_gridmap()

        plt.figure(1)
        plt.clf()
        plt.imshow(gridmap_image)
        plt.axis('off')
        plt.pause(0.00001)


class Config(object):
    def __init__(self):
        super(Config, self).__init__()

        self._set_action_dict()
        # self._set_orientation_dict()
        self._set_grid_dict()
        self._set_color_dict()

    def _set_action_dict(self):
        self.action_dict = {
            "stay": np.array([0, 0]),
            "move_up": np.array([-1, 0]),
            "move_down": np.array([1, 0]),
            "move_right": np.array([0, 1]),
            "move_left": np.array([+0, -1]),
            # "spin_right": +1,
            # "spin_left": -1,
        }

    # def _set_orientation_dict(self):
    #     self.orientation_dict = {
    #         "up": 0,
    #         "right": 1,
    #         "down": 2,
    #         "left": 3,
    #     }

    #     self.orientation_delta_dict = {
    #         "up": np.array([-1, 0]),
    #         "right": np.array([0, 1]),
    #         "down": np.array([1, 0]),
    #         "left": np.array([+0, -1]),
    #     }

    def _set_grid_dict(self):
        self.grid_dict = {
            "empty": 0,
            "wall": 1,
            "prey": 2,
            "predator": 3,
            # "orientation": 4,
        }

    def _set_color_dict(self):
        self.color_dict = {
            "empty": [0., 0., 0.],  # Black
            "wall": [0.5, 0.5, 0.5],  # Gray
            "prey": [1., 0., 0.],  # Red
            "predator": [0., 0., 1.],  # Blue
            "other_predator": [0., 1., 0.],  # Green
            # "other_predator": [0., 0., 0.5],  # Light Blue
            # "orientation": [0.1, 0.1, 0.1],  # Dark Gray
        }


class Agent(object):
    def __init__(self, i_agent, agent_type, base_gridmap_array):
        self.id = i_agent
        self.type = agent_type
        self.base_gridmap_array = base_gridmap_array

        self.config = Config()

        self._location = self._reset_location()
        # self._orientation = self.config.orientation_dict["up"]

    def _reset_location(self):
        location = np.array([
            np.random.choice(self.base_gridmap_array.shape[0]), 
            np.random.choice(self.base_gridmap_array.shape[1])])
        grid = self.base_gridmap_array[location[0], location[1]]

        while grid != self.config.grid_dict["empty"]:
            location = np.array([
                np.random.choice(self.base_gridmap_array.shape[0]), 
                np.random.choice(self.base_gridmap_array.shape[1])])
            grid = self.base_gridmap_array[location[0], location[1]]

        return location
        # if self.type == 'prey':
        #     loc_range_x = self.base_gridmap_array.shape[0]
        #     loc_range_y = np.arange(0, self.base_gridmap_array.shape[1] // 4)
        # elif self.type == 'predator':
        #     loc_range_x = self.base_gridmap_array.shape[0]
        #     loc_range_y = np.arange(3 * self.base_gridmap_array.shape[1] // 4,
        #                             self.base_gridmap_array.shape[1])
        
        # if self.type == 'prey':
        #     loc_range_y = np.arange(0,4)
        #     loc_range_x = np.arange(0,self.base_gridmap_array.shape[1])
        # elif self.type == 'predator':
        #     if self.id == 1:
        #         loc_range_y = np.arange(self.base_gridmap_array.shape[0] - 3,
        #                                 self.base_gridmap_array.shape[0] - 0)
        #         loc_range_x = np.arange(self.base_gridmap_array.shape[1] - 3,
        #                                 self.base_gridmap_array.shape[1] - 0)
        #     else:
        #         loc_range_y = np.arange(self.base_gridmap_array.shape[0] - 3,
        #                                 self.base_gridmap_array.shape[0] - 0)
        #         loc_range_x = np.arange(0,3)
        
        # if self.type == 'prey':
        #     loc_range_y = np.arange(0,1)
        #     loc_range_x = np.arange(self.base_gridmap_array.shape[1]//2 - 1,
        #                             self.base_gridmap_array.shape[1]//2 + 2)
        # elif self.type == 'predator':
        #     if self.id == 1:
        #         loc_range_y = np.arange(self.base_gridmap_array.shape[0] - 1,
        #                                 self.base_gridmap_array.shape[0] - 0)
        #         loc_range_x = np.arange(self.base_gridmap_array.shape[1] - 1,
        #                                 self.base_gridmap_array.shape[1] - 0)
        #     else:
        #         loc_range_y = np.arange(self.base_gridmap_array.shape[0] - 1,
        #                                 self.base_gridmap_array.shape[0] - 0)
        #         loc_range_x = np.arange(0,1)
            
        # location = np.array([
        #     np.random.choice(loc_range_x), 
        #     np.random.choice(loc_range_y)])
        # grid = self.base_gridmap_array[location[0], location[1]]
        
        # while grid != self.config.grid_dict["empty"]:
        #     location = np.array([
        #         np.random.choice(loc_range_x), 
        #         np.random.choice(loc_range_y)])
        #     grid = self.base_gridmap_array[location[0], location[1]]
        
        # return location

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        # Make sure inside grid
        if ((value[0] >= 0 and value[0] < self.base_gridmap_array.shape[0]) and
            (value[1] >= 0 and value[1] < self.base_gridmap_array.shape[1])):
            grid = self.base_gridmap_array[value[0], value[1]]
            # Make sure not on a wall
            if grid != self.config.grid_dict["wall"]:
                self._location = value

    # @property
    # def orientation(self):
    #     return self._orientation

    # @orientation.setter
    # def orientation(self, value):
    #     self._orientation = value % len(self.config.orientation_dict)

    # @property
    # def orientation_location(self):
    #     orientation = list(self.config.orientation_dict.keys())[self._orientation]
    #     orientation_delta = self.config.orientation_delta_dict[orientation]
    #     self._orientation_location = self.location + orientation_delta

    #     grid = self.base_gridmap_array[self._orientation_location[0], self._orientation_location[1]]
    #     if grid == self.config.grid_dict["wall"]:
    #         self._orientation_location = np.copy(self.location)

    #     return self._orientation_location


if __name__ == '__main__':
    N_PREDATOR = 2
    env = WolfpackCustomEnv(n_predator=N_PREDATOR)
    observations = env.reset() #; env.render()
    plt.figure(2) ; plt.cla() ; plt.imshow(observations[1]) ; plt.axis('off') ; plt.pause(2)
    # plt.figure(3) ; plt.cla() ; plt.imshow(observations[2]) ; plt.axis('off') ; plt.pause(0.00001)
    # while True:
    for _ in range(env.MAX_STEPS):
        prey_actions = [0]
        pred_actions = [env.action_space.sample() for _ in range(N_PREDATOR)]
        actions = prey_actions + pred_actions
        observations, rewards, done, _ = env.step(actions)
        plt.figure(2) ; plt.cla() ; plt.imshow(observations[1]) ; plt.axis('off') ; plt.pause(0.00001)
        # plt.figure(3) ; plt.cla() ; plt.imshow(observations[2]) ; plt.axis('off') ; plt.pause(0.00001)
        # env.render()
        if done:
            print(rewards, env.n_steps)
            break
        