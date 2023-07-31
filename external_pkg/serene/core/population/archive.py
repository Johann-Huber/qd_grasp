# Created by Giuseppe Paolo 
# Date: 27/07/2020

import os
import _pickle as pkl
import pdb

import numpy as np
from collections import deque

class Archive(object):
  """
  This class implements the archive. It only stores the BD and traj of each agent in unordered sets.
  We do not need to know anything else nor we need an order.
  """
  # ---------------------------------
  def __init__(self, parameters, name='archive'):
    self.data = deque() # This contains tuples with (agent_id, gt_bd bd, traj). The agent_id is necessary in case there are same bd-traj
    # self.data = set() # If wanna use set
    self.params = parameters
    self.stored_info = ['genome', 'bd', 'id', 'reward'] # ['genome', 'bd', 'id', 'reward', 'parent', 'born', 'stored', 'evaluated', 'ancestor', 'rew_area', 'gt_bd'] # self.params.archive_stored_info # Stuff is stored according to this order # TODO REQUIRED PARAMS
    # are those fields enough?
    self.name = name
    self.filled_tracker = []
  # ---------------------------------

  # ---------------------------------
  def __len__(self):
    """
    Returns the length of the archive
    """
    return self.size

  def __iter__(self):
    """
    Allows to directly iterate the pop.
    :return:
    """
    return self.data.__iter__()

  def __next__(self):
    """
    During iteration returns the next element of the iterator
    :return:
    """
    return self.data.__next__()

  def __getitem__(self, item):
    """
    Returns the asked item
    :param item: item to return. Can be an agent or a key
    :return: returns the corresponding agent or the column of the dict
    """
    if type(item) == str:
      try:
        index = self.stored_info.index(item) # Get index of item to get
        return [np.array(x[index]) for x in self.data] # Return list of those items
      except ValueError:
        raise ValueError('Wrong key given. Available: {} - Given: {}'.format(self.stored_info, item))
    else:
      return self.data[item]

  @property
  def size(self):
    """
    Size of the archive
    """
    return len(self.data)
  # ---------------------------------

  # ---------------------------------
  def store(self, agent):
    """
    Store data in the archive as a list of: (genome, gt_bd, bd, traj).
    No need to store the ID given that we store the genome.
    Saving as a tuple instead of a dict makes the append operation faster
    :param agent: agent to store
    :return:
    """
    # If want to use set
    # self.data.add((self._totuple(agent['genome']), self._totuple(agent['gt_bd']), self._totuple(agent['bd']),
    #                self._totuple(agent['traj'])))
    try:
      genome = np.array(agent).tolist()
      bd = agent.behavior_descriptor.values
      id = agent.gen_info.values['id']
      reward = agent.fitness.values[0]
      self.data.append([genome, bd, id, reward])
      #self.data.append([agent[info] for info in self.stored_info])
    except:
      pdb.set_trace()
    return True
  # ---------------------------------

  # ---------------------------------
  def _totuple(self, a):
    """
    This function converts trajectories to tuples to be added to the set
    :param a:
    :return:
    """
    try:
      return tuple(self._totuple(i) for i in a)
    except TypeError:
      return a
  # ---------------------------------

  # ---------------------------------
  def save(self, filepath, filename):
    """
    This function saves the population as a pkl file
    :param filepath:
    :return:
    """
    try:
      with open(os.path.join(filepath, '{}_{}.pkl'.format(self.name, filename)), 'wb') as file:
        pkl.dump([self.data, self.filled_tracker], file)
    except Exception as e:
      print('Cannot Save archive {}.'.format(filename))
      print('Exception {}'.format(e))
  # ---------------------------------

  # ---------------------------------
  def load(self, filepath):
    """
    This function loads the population
    :param filepath: File from where to load the population
    :return:
    """
    if not os.path.exists(filepath):
      print('File to load not found.')
      return

    if self.params is not None and self.params.verbose:
      print('Loading archive from {}'.format(filepath))
    with open(filepath, 'rb') as file:
      data = pkl.load(file)
    if len(data) == 2:
      self.filled_tracker = data[1]
      self.data = data[0]
    else:
      self.data = data # TODO remove this once all new exps are done
  # ---------------------------------

class Grid(Archive):
  def __init__(self, parameters, grid_params, name='archive'):
    """
    This data structure encodes the grid of MAP-Elites. For now only one agent per cell is supported
    :param parameters:
    :param grid_params:
    :param name:
    """
    super(Grid, self).__init__(parameters, name)
    self.grid_params = grid_params
    self.bd_dimensions = len(grid_params['max_coord'])

    self.grid = self.init_grid()
    self.filled_tracker = [] #Everytime a new cell is filled, the eval step is added to the list
    self.cell_lims = [np.linspace(grid_params['min_coord'][dim], grid_params['max_coord'][dim], num=grid_params['bins']+1) for dim in range(self.bd_dimensions)]

  def init_grid(self):
    """
    Initialized the grid with None values
    :return:
    """
    return np.full([self.grid_params['bins']] * self.bd_dimensions, fill_value=None)

  def store(self, agent):
    """
    Store data in the archive as a list of: (genome, gt_bd, bd, traj).
    No need to store the ID given that we store the genome.
    Saving as a tuple instead of a dict makes the append operation faster

    It also checks if the grid cell is already occupied. In case it is, saves the one with the highest fitness

    :param agent: agent to store
    :return:
    """
    assert len(agent['bd']) == self.bd_dimensions, print('BD of wrong size. Given: {} - Expected: {}'.format(len(agent['bd']), self.bd_dimensions))
    cell = self._find_cell(agent['bd'])
    if self.grid[cell] is None:
      # Add idx of agent in the datalist
      self.grid[cell] = agent['id']
      self.data.append([agent[info] for info in self.stored_info])
      self.filled_tracker.append(agent['evaluated'])
      return True
    else:
      # Find stored agent
      stored_agent_idx = self['id'].index(self.grid[cell])
      stored_agent = self.data[stored_agent_idx]
      # Only store if reward is higher
      if agent['reward'] >= stored_agent[self.stored_info.index('reward')]:
        del self.data[stored_agent_idx]
        self.grid[cell] = agent['id']
        self.data.append([agent[info] for info in self.stored_info])
        return True
      return False

  def _find_cell(self, bd):
    """
    This function finds in which cell the given BD belongs
    :param bd:
    :return:
    """
    cell_idx = []
    for dim in range(self.bd_dimensions):
      assert self.grid_params['max_coord'][dim] >= bd[dim] >= self.grid_params['min_coord'][dim], \
        print("BD outside of grid. BD: {} - Bottom Limits: {} - Upper Limits: {}".format(bd, self.grid_params['min_coord'], self.grid_params['max_coord']))

      # The max() is there so if we are at the bottom border the cell counts as the first
      cell_idx.append(max(np.argmax(self.cell_lims[dim] >= bd[dim]), 1) - 1) # Remove 1 for indexing starting at 0
    return tuple(cell_idx)

