import json
import random
import sys
from typing import Optional
from pprint import pprint
import time
import cProfile
import colorsys
from copy import deepcopy

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiBinary, Box

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType
from pettingzoo.test import api_test

from scipy.ndimage import gaussian_filter

class SlimeMultipleChem(AECEnv):
    def observe(self, agent: str) -> ObsType:
        return np.array(self.observations[agent])

    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
      
    def observations_n(self, same_obs=True):
        if same_obs:
            if isinstance(self.observation_space('0'), MultiBinary):
                return self.observation_space('0').n
            elif isinstance(self.observation_space('0'), Box):
                # observe maximum for agent's pheromone type and for the sum of all other types
                return self.observation_space('0').shape[0] ** 2

    def actions_n(self, same_actions=True):
        if same_actions:
            return self.action_space('0').n.item()

    metadata = {"render_modes": ["human", "server"]}

    def __init__(self,
                 seed,
                 render_mode: Optional[str] = None,
                 **kwargs):
        """
        :param population:          Controls the number of non-learning slimes (= green turtles)
        :param sniff_threshold:     Controls how sensitive slimes are to pheromone (higher values make slimes less
                                    sensitive to pheromone)—unclear effect on learning, could be negligible
        :param diffuse_area         Controls the diffusion radius
        :param diffuse_mode         Controls in which order patches with pheromone to diffuse are visited:
                                        'simple' = Python-dependant (dict keys "ordering")
                                        'rng' = random visiting
                                        'sorted' = diffuse first the patches with more pheromone
                                        'filter' = do not re-diffuse patches receiving pheromone due to diffusion
                                        'cascade' = step-by-step diffusion within 'diffuse_area'
        :param follow_mode          Controls how non-learning agents follow pheromone:
                                        'det' = follow greatest pheromone
                                        'prob' = follow greatest pheromone probabilistically (pheromone strength as weight)
        :param smell_area:          Controls the radius of the square area sorrounding the turtle whithin which it smells pheromone
        :param lay_area:            Controls the radius of the square area sorrounding the turtle where pheromone is laid
        :param lay_amount:          Controls how much pheromone is laid
        :param evaporation:         Controls how much pheromone evaporates at each step
        :param cluster_threshold:   Controls the minimum number of slimes needed to consider an aggregate within
                                    cluster-radius a cluster (the higher the more difficult to consider an aggregate a
                                    cluster)—the higher the more difficult to obtain a positive reward for being within
                                    a cluster for learning slimes
        :param cluster_radius:      Controls the range considered by slimes to count other slimes within a cluster (the
                                    higher the easier to form clusters, as turtles far apart are still counted together)
                                    —the higher the easier it is to obtain a positive reward for being within a cluster
                                    for learning slimes
        :param rew:                 Base reward for being in a cluster
        :param penalty:             Base penalty for not being in a cluster
        :param episode_ticks:       Number of ticks for episode termination
        :param W:                   Window width in # patches
        :param H:                   Window height in # patches
        :param PATCH_SIZE:          Patch size in pixels
        :param TURTLE_SIZE:         Turtle size in pixels
        :param FPS:                 Rendering FPS
        :param SHADE_STRENGTH:      Strength of color shading for pheromone rendering (higher -> brighter color)
        :param SHOW_CHEM_TEXT:      Whether to show pheromone amount on patches (when >= sniff-threshold)
        :param CLUSTER_FONT_SIZE:   Font size of cluster number (for overlapping agents)
        :param CHEMICAL_FONT_SIZE:  Font size of phermone amount (if SHOW_CHEM_TEXT is true)
        :param render_mode:
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        np.random.seed(seed)
        random.seed(seed)
        
        self.population = kwargs['population']
        self.learner_population = kwargs['learner_population']
        self.sniff_threshold = kwargs['sniff_threshold']
        self.diffuse_area = kwargs['diffuse_area']
        self.smell_area = kwargs['smell_area']
        self.lay_area = kwargs['lay_area']
        self.lay_amount = kwargs['lay_amount']
        self.evaporation = kwargs['evaporation']
        self.diffuse_mode = kwargs['diffuse_mode']
        self.follow_mode = kwargs['follow_mode']
        self.cluster_threshold = kwargs['cluster_threshold']
        self.cluster_radius = kwargs['cluster_radius']
        self.reward_type = kwargs['reward_type']
        self.reward = kwargs['rew']
        self.penalty = kwargs['penalty']
        self.episode_ticks = kwargs['episode_ticks']

        self.W = kwargs['W']
        self.H = kwargs['H']
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']

        self.coords = []
        self.offset = self.patch_size // 2
        self.W_pixels = self.W * self.patch_size
        self.H_pixels = self.H * self.patch_size
        for x in range(self.offset, (self.W_pixels - self.offset) + 1, self.patch_size):
            for y in range(self.offset, (self.H_pixels - self.offset) + 1, self.patch_size):
                self.coords.append((x, y))  # "centre" of the patch or turtle (also ID of the patch)

        pop_tot = self.population + sum(self.learner_population)
        self.possible_agents = [str(i) for i in range(self.population, pop_tot)]  # DOC learning agents IDs
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent = self._agent_selector.reset()

        n_coords = len(self.coords)

        # number of chemicals. It is the number of values specified in the self.learner_population list
        self.n_chemicals = len(self.learner_population)
        self.learners = dict()
        learner_id = 0
        for pheromone_type, n_learners in enumerate(self.learner_population):
            for _ in range(n_learners):
                self.learners[learner_id] = {"pos": self.coords[np.random.randint(n_coords)], "type": pheromone_type}
                learner_id += 1
        # create NON learner turtles
        self.turtles = {i: {"pos": self.coords[np.random.randint(n_coords)]} for i in range(self.population)}

        # patches-own [chemical] - amount of pheromone in each patch
        self.patches = {self.coords[i]: {"id": i,
                                         'chemical': np.zeros(self.n_chemicals),
                                         'turtles': []} for i in range(n_coords)}
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtles
        for t in self.turtles:
            self.patches[self.turtles[t]['pos']]['turtles'].append(t)

        # pre-compute relevant structures to speed-up computation during rendering steps
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed smell area for each patch, including itself
        self.smell_patches = self._find_neighbours(self.smell_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed lay area for each patch, including itself
        self.lay_patches = self._find_neighbours(self.lay_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed diffusion area for each patch, including itself
        if self.diffuse_mode == "cascade":
            assert isinstance(self.diffuse_area, int), "Error: diffuse_area must be an int"
            self.diffuse_patches = self._find_neighbours_cascade(self.diffuse_area)
        elif self.diffuse_mode in ("rng", "sorted", "filter", "rng-filter"):
            assert isinstance(self.diffuse_area, int), "Error: diffuse_area must be an int"
            self.diffuse_patches = self._find_neighbours(self.diffuse_area)

        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed cluster-check for each patch, including itself
        self.cluster_patches = self._find_neighbours(self.cluster_radius)

        self.actions = kwargs['actions']
        self._action_spaces = {
            a: Discrete(len(self.actions))
            for a in self.possible_agents
        }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        
        self.obs_type = kwargs['obs_type']
        # DOC obervation is an array of 8 real elements.
        # This array indicates the pheromone values in the 8 patches around the agent.
        if self.obs_type == "paper":
            self._observation_spaces = {
                a: Box(low=0.0, high=np.inf, shape=(8,self.n_chemicals), dtype=np.float32)
                for a in self.possible_agents
            }
        elif self.obs_type == "variation_1":
            self._observation_spaces = {
                a: MultiBinary(2)
                for a in self.possible_agents
            }  # DOC [0] = whether the turtle is in a cluster [1] = whether there is chemical in turtle patch

        #Different from AECEnv attribute self.rewards - only keeps last step rewards
        #self.rewards_cust = {i: [] for i in range(self.population, pop_tot)}
        #self.cluster_ticks = {i: 0 for i in range(self.population, pop_tot)}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.population, pop_tot)))
        )
    
    def get_learner_population(self):
        """
        Get the number of learners in this environment
        """
        return len(self.learners)

    def _find_neighbours_cascade(self, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area', 1 step at a time
        (visiting first 1-hop patches, then 2-hops patches, and so on)
        """

        neighbours = {}
        
        for p in self.patches:
            neighbours[p] = []
            for ring in range(area):
                for x in range(p[0] + (ring * self.patch_size), p[0] + ((ring + 1) * self.patch_size) + 1,
                               self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] + ((ring + 1) * self.patch_size) + 1,
                                   self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
                for x in range(p[0] + (ring * self.patch_size), p[0] - ((ring + 1) * self.patch_size) - 1,
                               -self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] - ((ring + 1) * self.patch_size) - 1,
                                   -self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
                for x in range(p[0] + (ring * self.patch_size), p[0] + ((ring + 1) * self.patch_size) + 1,
                               self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] - ((ring + 1) * self.patch_size) - 1,
                                   -self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
                for x in range(p[0] + (ring * self.patch_size), p[0] - ((ring + 1) * self.patch_size) - 1,
                               -self.patch_size):
                    for y in range(p[1] + (ring * self.patch_size), p[1] + ((ring + 1) * self.patch_size) + 1,
                                   self.patch_size):
                        if (x, y) not in neighbours[p]:
                            neighbours[p].append((x, y))
            neighbours[p] = [self._wrap(x, y) for (x, y) in neighbours[p]]

        return neighbours

    def _find_neighbours(self, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area'
        """

        neighbours = {}
        
        for p in self.patches:
            neighbours[p] = []
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            neighbours[p] = list(set(neighbours[p]))

        return neighbours

    def _wrap(self, x: int, y: int):
        """
        Wrap x,y coordinates around the torus

        :param x: the x coordinate to wrap
        :param y: the y coordinate to wrap
        :return: the wrapped x, y
        """
        return x % self.W_pixels, y % self.H_pixels

    def _find_pheromone_gradient(self, agent, ph_type="own", max_grad=True):
        """
        Finds the patch with greatest gradient with respect to the agent's position

        :param ph_type: "own" will return the max corresponding to agent's pheromone type; "other" will return the max of the sum of all the other types of pheromones, except the agent's one
        :param max_grad: True if we wish to follow the gradient (go towards max), False otherwise.
        """
        turtle_ph_type = self.learners[agent]['type']
        pos = self.learners[agent]['pos']
        match ph_type:
            case "own":
                pheromone_quantity = lambda ph: ph[turtle_ph_type]
            case "other":
                pheromone_quantity = lambda ph: np.sum(np.delete(ph, turtle_ph_type))
            case _:
                raise ValueError(f"Invalid pheromone selector: {ph_type}")
        smell_patches_here = self.smell_patches[pos]
        np_rng = np.random.default_rng()
        pheromones = np.array([pheromone_quantity(self.patches[p]['chemical']) for p in smell_patches_here])
        if self.follow_mode == "prob":
            if np.all(pheromones == 0.):
                weights = None
            else:
                # very simple: if we want to follow the minimum, just take the complementary probabilities
                weights = (-1. * max_grad)*(pheromones / np.sum(pheromones)) + (1.0 * max_grad)
            selected = np_rng.choice(len(smell_patches_here), p=weights)
            max_ph = pheromones[selected]
            winner_patch = smell_patches_here[selected]
        elif self.follow_mode == "det":
            if max_grad:
                grad_op = np.argmax
            else:
                grad_op = np.argmin
            selected = grad_op(pheromones)
            max_ph = pheromones[selected]
            winner_patch = smell_patches_here[selected]
        else:
            raise ValueError(f"Invalid follow_mode: {self.follow_mode}")
        return max_ph, winner_patch

    
    def _take_action(self, agent, action):
        action_str = self.actions[action]

        match action_str:
            case 'walk':
                self._walk(agent)
            case 'lay_pheromone':
                self._lay_pheromone(agent)
            case 'follow_own_pheromone':
                self._follow_own_pheromone(agent)
            case 'run_away_own_pheromone':
                self._run_away_own_pheromone(agent)
            case 'follow_other_pheromone':
                self._follow_other_pheromone(agent)
            case 'run_away_other_pheromone':
                self._run_away_other_pheromone(agent)
            case _:
                raise ValueError("Action out of range!")
    
    def _move_agent(self, agent, target_patch, pheromone):
        if pheromone >= self.sniff_threshold:
            turtle = self.learners[agent]
            self.patches[turtle['pos']]['turtles'].remove(agent)
            turtle['pos'] = target_patch
            self.patches[target_patch]['turtles'].append(agent)

    def _walk(self, agent):
        """
        Move in random direction (8 sorrounding cells)
        """
        turtle = self.learners[agent]
        choice = [self.patch_size, -self.patch_size, 0]
        x, y = turtle['pos']
        self.patches[turtle['pos']]['turtles'].remove(agent)
        x_rnd = np.random.choice(choice)
        y_rnd = np.random.choice(choice)
        x2, y2 = x + x_rnd, y + y_rnd
        x2, y2 = self._wrap(x2, y2)
        
        turtle['pos'] = (x2, y2)
        self.patches[turtle['pos']]['turtles'].append(agent)
    
    def _lay_pheromone(self, agent):
        """
        Lay 'lay_amount' pheromone of type 'type' in square 'area' centred in 'pos'
        """
        ph_type = self.learners[agent]['type']
        pos = self.learners[agent]['pos']
        for p in self.lay_patches[pos]:
            self.patches[p]['chemical'][ph_type] += self.lay_amount

    def _follow_own_pheromone(self, agent):
        """
        Follow the scent of my pheromone type, ignoring the others
        """
        pheromone, target_patch = self._find_pheromone_gradient(agent)
        self._move_agent(agent, target_patch, pheromone)

    def _run_away_own_pheromone(self, agent):
        """
        Run away from the scent of this agent's pheromone type, ignoring the others
        """
        pheromone, target_patch = self._find_pheromone_gradient(agent, "own", False)
        self._move_agent(agent, target_patch, pheromone)
    
    def _follow_other_pheromone(self, agent):
        """
        Follow the scent of pheromone types different from this agent's one. We consider the sum of all the other pheromones, not just one in particular
        """
        pheromone, target_patch = self._find_pheromone_gradient(agent, "other")
        self._move_agent(agent, target_patch, pheromone)

    def _run_away_other_pheromone(self, agent):
        """
        Run away from the scent of pheromone types different from this agent's one. We consider the sum of all the other pheromones, not just one in particular
        """
        pheromone, target_patch = self._find_pheromone_gradient(agent, "other", False)
        self._move_agent(agent, target_patch, pheromone)


    # learners act
    def step(self, action: int):
        if(self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        self.agent = self.agent_name_mapping[self.agent_selection]  # ID of agent

        # Non dovrei calcolare il reward e le osservazioni dopo aver fatto un'azione???
        self.observations[str(self.agent)], self.cluster_ticks, self.rewards_cust = self.process_agent(
            self.cluster_ticks,
            self.rewards_cust,
        )
        
        self._take_action(self.agent, action)

        if self._agent_selector.is_last():
            for ag in self.agents:
                self.rewards[ag] = self.rewards_cust[self.agent_name_mapping[ag]][-1]
            if len(self.turtles) > 0:
                self.turtles, self.patches = self.move(self.turtles, self.patches)
            self.patches = self._diffuse_and_evaporate(self.patches)
        else:
            self._clear_rewards()
            
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[str(self.agent)] = 0
        self._accumulate_rewards()

    def process_agent(self, cluster_ticks, rewards_cust):
        """
        In this methods we compute the agent's reward and it's observation.
        """
        cluster = self._compute_cluster(self.agent)

        if self.reward_type == "cluster":
            cluster_ticks, rewards_cust, cur_reward = self.reward_cluster_and_time_punish_time(
                cluster_ticks,
                rewards_cust,
                cluster
            )
        elif self.reward_type == "scatter":
            cluster_ticks, rewards_cust, cur_reward = self.reward_scatter_and_time_punish_time(
                cluster_ticks,
                rewards_cust,
                cluster
            )
        
        if self.obs_type == "paper":
            #_, max_coords = self._find_max_pheromone(self.learners[self.agent]['pos'])
            #observations = np.array(max_coords)
            observations = self._get_obs(self.learners[self.agent])
        elif self.obs_type == "variation_1":
            chemical = self._check_chemical(self.agent)
            observations = np.array([cluster >= self.cluster_threshold, chemical])

        return observations, cluster_ticks, rewards_cust

    def _get_obs(self, agent):
        """
        This method return the observation give by the env.
        The array indicates the pheromone values in the 8 patches around the agent.
        """
        pos = agent['pos']
        field_of_view = [
            self._wrap(r, c)
            for r in range(pos[0] - self.patch_size, pos[0] + 2 * self.patch_size, self.patch_size)
            for c in range(pos[1] - self.patch_size, pos[1] + 2 * self.patch_size, self.patch_size)
        ]
        field_of_view.remove(pos)
        obs = np.array([self.patches[f]["chemical"] for f in field_of_view])
        return obs

    def convert_observation(self, obs, agent):
        """
        This method returns the conversion of the observation to an integer.
        It's useful for IQL.
        """
        if self.obs_type == "paper":
            rng = np.random.default_rng()
            chem_type = self.learners[agent]['type']
            n_patches = obs.shape[0]    # number of patches observed
            obs = obs.transpose(1,0)
            own_chem_obs = obs[chem_type]
            other_chem_obs = np.sum(np.delete(obs, chem_type, axis=0), axis=0)
            if np.unique(own_chem_obs).shape[0] == 1:
                max_own = rng.integers(n_patches)
            else:
                max_own = np.argmax(own_chem_obs).item()
            if np.unique(other_chem_obs).shape[0] == 1:
                max_other = rng.integers(n_patches)
            else:
                max_other = np.argmax(other_chem_obs).item()
                # found the id of the cell with max of "own" pheromone and id of cell with max of "other" -> get single id by "flattening" to the 64 possible combinations
            obs_id = max_own + n_patches*max_other
        elif self.obs_type == "variation_1":
            obs_id = int(f"{obs[0].astype(np.uint8)}{obs[1].astype(np.uint8)}", 2)
        return obs_id

    def lay_pheromone(self, patches, pos, type):
        """
        Lay 'amount' pheromone of type 'type' in square 'area' centred in 'pos'
        """
        for p in self.lay_patches[pos]:
            patches[p]['chemical'][type] += self.lay_amount
        
        return patches

    def _diffuse(self, patches):
        """
        Diffuses pheromone from each patch to nearby patches controlled through self.diffuse_area patches in a way
        controlled through self.diffuse_mode:
            'simple' = Python-dependant (dict keys "ordering")
            'rng' = random visiting
            'sorted' = diffuse first the patches with more pheromone
            'filter' = do not re-diffuse patches receiving pheromone due to diffusion
        """
        n_size = len(self.diffuse_patches[list(patches.keys())[0]])  # same for every patch
        patch_keys = list(patches.keys())
        
        if self.diffuse_mode == 'rng':
            random.shuffle(patch_keys)
        elif self.diffuse_mode == 'sorted':
            patch_list = list(patches.items())
            # sum over all types of chemicals (valid also in other modes)
            patch_list = sorted(patch_list, key=lambda t: np.sum(t[1]['chemical']), reverse=True)
            patch_keys = [t[0] for t in patch_list]
        elif self.diffuse_mode == 'filter':
            patch_keys = [k for k in patches if np.sum(patches[k]['chemical']) > 0]
        elif self.diffuse_mode == 'rng-filter':
            patch_keys = [k for k in patches if np.sum(patches[k]['chemical']) > 0]
            random.shuffle(patch_keys)
        
        for patch in patch_keys:
            p = patches[patch]['chemical']
            ratio = p / n_size
            
            if p > 0:
                diffuse_keys = self.diffuse_patches[patch][:]
                
                for n in diffuse_keys:
                    patches[n]['chemical'] += ratio
                
                patches[patch]['chemical'] = ratio

        return patches

    def _gaussian_diffusion(self, patches_array):
        """
        Computes gaussian diffusion on a set of patches.
        patches_array is a numpy array with shape (n_patches, n_chems)
        n_patches -> number of patches to diffuse
        n_chems -> number of pheromones
        """
        n_patches, _ = patches_array.shape
        grid = patches_array.reshape((self.W, self.H, -1)).transpose(2,1,0)
        grid = gaussian_filter(grid, sigma=(self.diffuse_area, self.diffuse_area), mode=("wrap","wrap"), axes=(1,2))
        grid = grid.transpose(2,1,0).reshape((n_patches, -1))
        return grid


    def _diffuse2(self, patches):
        """
        This diffuse method use a gaussian filter for the process.
        This is a kind of parallel diffusion.
        """
        grid = np.array([patches[p]["chemical"] for p in patches.keys()])
        # perform gaussian diffusion
        grid = self._gaussian_diffusion(grid)
        for p, g in zip(patches, grid):
            patches[p]['chemical'] = g
        return patches

    def _evaporate(self, patches):
        """
        Evaporates pheromone from each patch according to param self.evaporation
        """
        for patch in patches.keys():
            #if patches[patch]['chemical'] > 0:
            # just works with multiple pheromones thanks to np array
            patches[patch]['chemical'] *= self.evaporation
        return patches

    def _diffuse_and_evaporate(self, patches):
        """
        This method combine the _diffuse2 and _evaporate methods in one function.
        It is the method currently used.
        """
        # Diffusion
        grid = np.array([patches[p]["chemical"] for p in patches.keys()])
        grid = self._gaussian_diffusion(grid)
        # Evaporation
        grid *= self.evaporation
        # Write values
        for p, g in zip(patches, grid):
            patches[p]['chemical'] = g
        
        return patches

    def walk(self, patches, turtle):
        """
        Action 0: move in random direction (8 sorrounding cells)
        """      
        choice = [self.patch_size, -self.patch_size, 0]
        x, y = turtle['pos']
        patches[turtle['pos']]['turtles'].remove(self.agent)
        x_rnd = np.random.choice(choice)
        y_rnd = np.random.choice(choice)
        x2, y2 = x + x_rnd, y + y_rnd
        x2, y2 = self._wrap(x2, y2)
        
        turtle['pos'] = (x2, y2)
        patches[turtle['pos']]['turtles'].append(self.agent)

        return patches, turtle

    def run_away_pheromone(self, patches, ph_coords, turtle):
        """
        Action 3: don't follow/avoid the pheromone.
        """
        x, y = turtle['pos']
        patches[turtle['pos']]['turtles'].remove(self.agent)
        if ph_coords[0] > x and ph_coords[1] > y:
            x -= self.patch_size
            y -= self.patch_size
        elif ph_coords[0] < x and ph_coords[1] < y:
            x += self.patch_size
            y += self.patch_size
        elif ph_coords[0] > x and ph_coords[1] < y:
            x -= self.patch_size
            y += self.patch_size
        elif ph_coords[0] < x and ph_coords[1] > y:
            x += self.patch_size
            y -= self.patch_size
        elif ph_coords[0] == x and ph_coords[1] < y:
            choices = [self.patch_size, -self.patch_size]
            x += random.choice(choices)
            y += self.patch_size
        elif ph_coords[0] == x and ph_coords[1] > y:
            choices = [self.patch_size, -self.patch_size]
            x += random.choice(choices)
            y -= self.patch_size
        elif ph_coords[0] > x and ph_coords[1] == y:  
            choices = [self.patch_size, -self.patch_size]
            x -= self.patch_size
            y += random.choice(choices)
        elif ph_coords[0] < x and ph_coords[1] == y:   
            choices = [self.patch_size, -self.patch_size]
            x += self.patch_size
            y += random.choice(choices)
        else:  # my patch
            choices = [self.patch_size, -self.patch_size]
            x += random.choice(choices)
            y += random.choice(choices)
        x, y = self._wrap(x, y)
        turtle['pos'] = (x, y)
        patches[turtle['pos']]['turtles'].append(self.agent)

        return patches

    def follow_pheromone(self, patches, ph_coords, turtle):
        """
        Action 2: move turtle towards greatest pheromone found
        """
        x, y = turtle['pos']
        patches[turtle['pos']]['turtles'].remove(self.agent)
        if ph_coords[0] > x and ph_coords[1] > y:  # top right
            x += self.patch_size
            y += self.patch_size
        elif ph_coords[0] < x and ph_coords[1] < y:  # bottom left
            x -= self.patch_size
            y -= self.patch_size
        elif ph_coords[0] > x and ph_coords[1] < y:  # bottom right
            x += self.patch_size
            y -= self.patch_size
        elif ph_coords[0] < x and ph_coords[1] > y:  # top left
            x -= self.patch_size
            y += self.patch_size
        elif ph_coords[0] == x and ph_coords[1] < y:  # below me
            y -= self.patch_size
        elif ph_coords[0] == x and ph_coords[1] > y:  # above me
            y += self.patch_size
        elif ph_coords[0] > x and ph_coords[1] == y:  # right
            x += self.patch_size
        elif ph_coords[0] < x and ph_coords[1] == y:  # left
            x -= self.patch_size
        else:  # my patch
            pass
        x, y = self._wrap(x, y)
        turtle['pos'] = (x, y)
        patches[turtle['pos']]['turtles'].append(self.agent)

        return patches

    def _find_max_pheromone(self, pos, type):
        """
        Find where the maximum pheromone level is within a square controlled by self.smell_area centred in 'pos'.
        Following pheromone modeis controlled by param self.follow_mode:
            'det' = follow greatest pheromone
            'prob' = follow greatest pheromone probabilistically (pheromone strength as weight)
        """
        if self.follow_mode == "prob":
            population = [k for k in self.smell_patches[pos]]
            weights = [self.patches[k]['chemical'][type] for k in self.smell_patches[pos]]
            if all([w == 0 for w in weights]):
                winner = population[np.random.choice(len(population))]
            else:
                winner = random.choices(population, weights=weights, k=1)[0]
            max_ph = self.patches[winner]['chemical'][type]
        else:
            max_ph = -1
            max_pos = [pos]
            for p in self.smell_patches[pos]:
                chem = self.patches[p]['chemical'][type]
                if chem > max_ph:
                    max_ph = chem
                    max_pos = [p]
                elif chem == max_ph:
                    max_pos.append(p)
            winner = max_pos[np.random.choice(len(max_pos))]

        return max_ph, winner

    def _compute_cluster(self, current_agent):
        """
        Checks whether the learner turtle is within a cluster, given 'cluster_radius' and 'cluster_threshold'
        """
        cluster = 0
        for p in self.cluster_patches[self.learners[current_agent]['pos']]:
            cluster += len(self.patches[p]['turtles'])

        return cluster

    #def avg_cluster(self):
    #    """
    #    Record the cluster size. It's a fuzzy computation.
    #    """
    #    cluster_sizes = []  # registra la dim. dei cluster
    #    for l in self.learners:
    #        cluster = []  # tiene conto di quali turtle sono in quel cluster
    #        for p in self.cluster_patches[self.learners[l]['pos']]:
    #            for t in self.patches[p]['turtles']:
    #                cluster.append(t)
    #        cluster.sort()
    #        if cluster not in cluster_sizes:
    #            cluster_sizes.append(cluster)

    #    # cleaning process: confornta i cluster (nello stesso episodio) e se ne trova 2 con più del 90% di turtle uguali ne elimina 1
    #    for cluster in cluster_sizes:
    #        for cl in cluster_sizes:
    #            if cl != cluster:
    #                intersection = list(set(cluster) & set(cl))
    #                if len(intersection) > len(cluster) * 0.90:
    #                    cluster_sizes.remove(cl)

    #    # calcolo avg_cluster_size
    #    somma = 0
    #    for cluster in cluster_sizes:
    #        somma += len(cluster)
    #    avg_cluster_size = somma / len(cluster_sizes)
    #    return avg_cluster_size
    
    def avg_cluster2(self):
        """
        Same compuation as avg_cluster.
        Use THIS for calculating the average, avg_cluster has a bug!
        """
        cluster_sizes = {i: list() for i in range(self.n_chemicals)}  # record cluster dim by type of chemical
        for l in self.learners:
            cluster = {i: list() for i in range(self.n_chemicals)}  # tiene conto di quali turtle sono in quel cluster
            for p in self.cluster_patches[self.learners[l]['pos']]:
                for t in self.patches[p]['turtles']:
                    cluster[self.learners[t]['type']].append(t)
            #cluster.sort()
            for typ, cl in cluster.items():
                if cl not in cluster_sizes[typ]:
                    cluster_sizes[typ].append(cl)
        
        cs = deepcopy(cluster_sizes)
        for cluster_type, cl in cluster_sizes.items():
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    set1 = set(cl[j])
                    set2 = set(cl[i])
                    if set1.issubset(set2) and cl[j] in cs[cluster_type]:
                        cs[cluster_type].remove(cl[j])
                    elif set2.issubset(set1) and cl[i] in cs[cluster_type]:
                        cs[cluster_type].remove(cl[i])
        # calcolo avg_cluster_size
        avg_cluster_size = {typ: 0.0 for typ in cs.keys()}
        for typ, cluster in cs.items():
            for cl in cluster:
                avg_cluster_size[typ] += len(cl)
            avg_cluster_size[typ] /= len(cluster)
        return avg_cluster_size

    def _check_chemical(self, current_agent):
        """
        Checks whether there is pheromone on the patch where the learner turtle is
        """
        type = self.learners[current_agent]['type']
        return self.patches[self.learners[current_agent]['pos']][
                'chemical'][type] > self.sniff_threshold

    # not a real reward function
    def test_reward(self, current_agent):  # trying to invert rewards process, GOAL: check any strange behaviour
        """
        :return: the reward
        """
        self.agent = current_agent
        chem = 0
        for p in self.patches.values():
            if self.agent in p['turtles']:
                chem = p['chemical']
        if chem >= 5:
            cur_reward = -1000
        else:
            cur_reward = 100

        self.rewards_cust[self.agent].append(cur_reward)
        return cur_reward

    def reward_cluster_punish_time(self, current_agent):  # DOC NetLogo rewardFunc7
        """
        Reward is (positve) proportional to cluster size (quadratic) and (negative) proportional to time spent outside
        clusters
        """
        self.agent = current_agent
        cluster = self._compute_cluster(self.agent)
        if cluster >= self.cluster_threshold:
            self.cluster_ticks[self.agent] += 1

        cur_reward = ((cluster ^ 2) / self.cluster_threshold) * self.reward + (
                ((self.episode_ticks - self.cluster_ticks[self.agent]) / self.episode_ticks) * self.penalty)

        self.rewards_cust[self.agent].append(cur_reward)
        return cur_reward

    def reward_cluster_and_time_punish_time(self, cluster_ticks, rewards_cust, cluster):
        """
        The clustering reward used in the article.
        """
        if cluster >= self.cluster_threshold:
            cluster_ticks[self.agent] += 1

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.reward + \
                     (cluster / self.cluster_threshold) * (self.reward ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.penalty)

        rewards_cust[self.agent].append(cur_reward)
        return cluster_ticks, rewards_cust, cur_reward

    def reward_cluster_and_time_punish_time(self, cluster_ticks, rewards_cust, cluster, n_others):
        """
        Clustering reward that gives a penalty based on number of agents of different type in neighbourhood
        """
        if cluster >= self.cluster_threshold:
            cluster_ticks[self.agent] += 1

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.reward + \
                     (cluster / self.cluster_threshold) * (self.reward ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.penalty) + \
                     n_others * self.penalty

        rewards_cust[self.agent].append(cur_reward)
        return cluster_ticks, rewards_cust, cur_reward
    
    def reward_scatter_and_time_punish_time(self, cluster_ticks, rewards_cust, cluster):
        """
        The scattering reward used in the article.
        """
        if cluster >= self.cluster_threshold:
            cluster_ticks[self.agent] += 1

        cur_reward = (cluster_ticks[self.agent] / self.episode_ticks) * self.penalty + \
                     (cluster / self.cluster_threshold) * (self.penalty ** 2) + \
                     (((self.episode_ticks - cluster_ticks[self.agent]) / self.episode_ticks) * self.reward)

        rewards_cust[self.agent].append(cur_reward)
        return cluster_ticks, rewards_cust, cur_reward

    def reset(self, seed=None, return_info=True, options=None):
        """
        Reset env.
        """
        # empty stuff
        pop_tot = self.population + sum(self.learner_population)
        self.rewards_cust = {i: [] for i in range(self.population, pop_tot)}
        self.cluster_ticks = {i: 0 for i in range(self.population, pop_tot)}
        
        #Initialize attributes for PettingZoo Env
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        
        # re-position learner turtle
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].remove(l)
            self.learners[l]['pos'] = self.coords[np.random.randint(len(self.coords))]
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle
        # re-position NON learner turtles
        for t in self.turtles:
            self.patches[self.turtles[t]['pos']]['turtles'].remove(t)
            self.turtles[t]['pos'] = self.coords[np.random.randint(len(self.coords))]
            self.patches[self.turtles[t]['pos']]['turtles'].append(t)
        # patches-own [chemical] - amount of pheromone in the patch
        for p in self.patches:
            self.patches[p]['chemical'][:] = 0.0

        if self.obs_type == "paper":
            #self.observations = {
            #    a: np.array(self.learners[int(a)]['pos'])
            #    for a in self.agents
            #}
            #self.observations = {
            #    a: np.random.randint(8, dtype=np.int32)
            #    for a in self.agents
            #}
            self.observations = {
                a: np.zeros((8, self.n_chemicals), dtype=np.float32)
                for a in self.agents
            }
        elif self.obs_type == "variation_1":
            self.observations = {a: np.full((2, ), False) for a in self.agents}
        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def get_neighborood_chemical(self, agent, as_vectors=False):
        agent_pos = self.learners[agent]["pos"]
        smell_patches = self.smell_patches[agent_pos]
        
        output_mask = []
        for patch in smell_patches:
            output_mask.append(self.patches[patch]["chemical"] - self.patches[agent_pos]["chemical"]) if as_vectors else output_mask.append(self.patches[patch]["chemical"])

        return np.array([output_mask], dtype=np.float32)


import pygame
import colorsys

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (190, 0, 0)
GREEN = (0, 190, 0)

class SlimeVisualizer:
    def __init__(
        self,
        W_pixels,
        H_pixels,
        **kwargs
    ):
        self.fps = kwargs['FPS']
        self.shade_strength = kwargs['SHADE_STRENGTH']
        self.show_chem_text = kwargs['SHOW_CHEM_TEXT']
        self.cluster_font_size = kwargs['CLUSTER_FONT_SIZE']
        self.chemical_font_size = kwargs['CHEMICAL_FONT_SIZE']
        #self.sniff_threshold = kwargs['sniff_threshold']
        self.sniff_threshold = 0.0 
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']

        self.W_pixels = W_pixels
        self.H_pixels = H_pixels
        self.offset = self.patch_size // 2
        self.screen = pygame.display.set_mode((self.W_pixels, self.H_pixels))
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.cluster_font = pygame.font.SysFont("arial", self.cluster_font_size)
        self.chemical_font = pygame.font.SysFont("arial", self.chemical_font_size)
        self.first_gui = True
        # sample one color for each chemical. also for agents
        n_chems = kwargs['N_CHEMICALS']
        rnd = np.random.default_rng()
        # build chem_colors in HLS [0,1] float. start with pure hues -> select random hues
        self.chem_colors = rnd.random(n_chems)
        self.agent_colors = [tuple(random.randint(10, 255) for _ in range(3)) for _ in range(n_chems)]

    def render(
        self,
        patches,
        learners,
        turtles,
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # window closed -> program quits
                pygame.quit()

        if self.first_gui:
            self.first_gui = False
            pygame.init()
            pygame.display.set_caption("SLIME")

        self.screen.fill(BLACK)
        # draw patches
        for p in patches:
            patch_color = self._compute_patch_color(patches[p]['chemical'])
            pygame.draw.rect(
                self.screen,
                patch_color,
                pygame.Rect(
                    p[0] - self.offset,
                    p[1] - self.offset,
                    self.patch_size,
                    self.patch_size
                )
            )
        # draw learners
        for learner in learners.values():
            pygame.draw.circle(self.screen, self.agent_colors[learner['type']], (learner['pos'][0], learner['pos'][1]), self.turtle_size // 2)
        # draw NON learners
        for turtle in turtles.values():
            pygame.draw.circle(self.screen, BLUE, (turtle['pos'][0], turtle['pos'][1]), self.turtle_size // 2)

        for p in patches:
            if len(patches[p]['turtles']) > 1:
                text = self.cluster_font.render(str(len(patches[p]['turtles'])), True,
                                                RED if -1 in patches[p]['turtles'] else WHITE)
                self.screen.blit(text, text.get_rect(center=p))

        self.clock.tick(self.fps)
        pygame.display.flip()

        return pygame.surfarray.array3d(self.screen)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _compute_patch_color(self, pheromones):
        """
        Computes color of a patch based on the quantity of each pheromone
        """
        tot_pher = np.sum(pheromones)
        if tot_pher > 0.0:
            # get pheromone with highest quantity
            highest_p = np.argmax(pheromones)
            # saturation should be zero when all pheromones are the same
            scaled_saturation = pheromones[highest_p] / tot_pher - 1./pheromones.shape[0]
            selected_hue = self.chem_colors[highest_p]
            # scale value based on total amount of pheromone
            scaled_value = np.clip(tot_pher / self.shade_strength, 0., 1.)
        else:
            # no pheromone -> black
            selected_hue = 0.0
            scaled_value = 0.0
            scaled_saturation = 0.0
        rgb = colorsys.hsv_to_rgb(selected_hue, scaled_saturation, scaled_value)
        return (int(round(rgb[0]*255)), int(round(rgb[1]*255)), int(round(rgb[2]*255)))

def main():
    params = {
        "population": 0,
        #"learner_population": 50,
        "learner_population": 25,
        "actions": [
            #"move-toward-chemical",
            "random-walk",
            "drop-chemical",
            #"move-and-drop",
            #"walk-and-drop",
            #"move-away-chemical"
        ],
        "sniff_threshold": 0.9,
        "diffuse_area": 0.5,
        "diffuse_mode": "gaussian",
        #"diffuse_mode": "cascade",
        "follow_mode": "prob",
        "smell_area": 1,
        "lay_area": 0,
        "lay_amount": 3,
        "evaporation": 1,
        "cluster_threshold": 30,
        "cluster_radius": 5,
        #"obs_type": "variation_1",
        "obs_type": "paper",
        "reward_type": "scatter",
        "rew": 100,
        "penalty": -1,
        #"episode_ticks": 500,
        "episode_ticks": 1000,
        #"W": 66,
        "W": 25,
        #"H": 38,
        "H": 25,
        "PATCH_SIZE": 20,
        #"PATCH_SIZE": 10,
        "TURTLE_SIZE": 16,
        #"TURTLE_SIZE": 8,
    }

    params_visualizer = {
      "FPS": 15,
      #"FPS": 3,
      "SHADE_STRENGTH": 10,
      "SHOW_CHEM_TEXT": True,
      "CLUSTER_FONT_SIZE": 12,
      "CHEMICAL_FONT_SIZE": 8,
      "gui": True,
      "sniff_threshold": 0.9,
      "PATCH_SIZE": 20,
      #"PATCH_SIZE": 10,
      "TURTLE_SIZE": 16,
      #"TURTLE_SIZE": 8,
    }

    from tqdm import tqdm

    EPISODES = 50
    LOG_EVERY = 1
    SEED = 0
    np.random.seed(SEED)
    env = SlimeMultipleChem(SEED, **params)
    env_vis = SlimeVisualizer(env.W_pixels, env.H_pixels, **params_visualizer)
    #actions = [1, 3]
    ACTION_NUM = len(params["actions"])

    start_time = time.time()
    for ep in tqdm(range(1, EPISODES + 1), desc="Episode"):
        env.reset()
        for tick in tqdm(range(params['episode_ticks']), desc="Tick", leave=False):
            for agent in env.agent_iter(max_iter=params["learner_population"]):
                observation, reward, _ , _, info = env.last(agent)
                #action = np.random.randint(0, ACTION_NUM)
                action = 1
                #action = random.choice(actions)
                env.step(action)
            env_vis.render(
                env.patches,
                env.learners,
                env.turtles
            )

    print("Total time = ", time.time() - start_time)
    env.close()

if __name__ == "__main__":
    #PARAMS_FILE = "multi-agent-env-params.json"
    main()
    #cProfile.run("main()", "possible_refactor_min.prof")
