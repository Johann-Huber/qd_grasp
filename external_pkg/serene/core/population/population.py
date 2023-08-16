# Created by Giuseppe Paolo 
# Date: 27/07/2020

import os
import pdb

import numpy as np
import _pickle as pkl

import utils.constants as consts


class Population(object):
  """
  Population class based on a dict
  """

  # ---------------------------------
  def __init__(self, parameters, init_size=None, name='population'):
    """
    Constructor. Takes as input the parameters
    :param parameters:
    """

    self.pop = []
    self.agent_id = 0
    self.name = name
    self.agent_template = consts.serene_agent_template

    self.genome_size = parameters['genome_size']
    self.genome_limit = parameters['genome_limit']
    if init_size is None:
      self.init_pop_size = parameters['pop_size']
    else:
      self.init_pop_size = init_size
    for k in range(self.init_pop_size):
      self.add()
  # ---------------------------------

  # ---------------------------------
  # These function make the class indexable and iterable like a list
  def __iter__(self):
    """
    Allows to directly iterate the pop.
    :return:
    """
    return self.pop.__iter__()

  def __next__(self):
    """
    During iteration returns the next element of the iterator
    :return:
    """
    return self.pop.__next__()

  def __getitem__(self, item):
    """
    Returns the asked property
    :param item: item to return. Can be an agent or a key
    :return: returns the corresponding agent or the column of the dict
    """
    if type(item) == str:
      assert item in list(self.agent_template.keys()), 'Wrong key given. Available: {} - Given: {}'.format(list(self.agent_template.keys()), item)
      return [agent[item] for agent in self.pop]
    else:
      return self.pop[item]

  def __setitem__(self, key, value):
    """
    Set the agent in position key with the ones passed as value
    :param key: if int: position of the agent to set. if something else: column of the dict to update
    :param value: New agent to set or list of elements to update
    :return:
    """
    if type(key) == str:
      assert key in list(self.agent_template.keys()), 'Wrong key given. Available: {} - Given: {}'.format(list(self.agent_template.keys()), key)
      assert len(value) == self.size, 'List of values different from pop size. N of values: {} - Pop size: {}'.format(len(value), self.size)
      for k in range(self.size):
        self.pop[k][key] = value[k]
    else:
      assert self.size > key > -self.size - 1, 'Index out of range'
      self.pop[key] = value

  def __len__(self):
    """
    Returns the length of the population
    """
    return self.size

  @property
  def size(self):
    """
    Size of the population
    """
    return len(self.pop)
  # ---------------------------------

  # ---------------------------------
  def add(self, agent=None):
    """
    Adds agent to population.
    :param agent: Agent to add. If None generates a new agent
    :return:
    """
    if agent is None:
      agent = self.agent_template.copy()
      agent['id'] = self.agent_id
      agent['genome'] = self.generate_gen()
      self.agent_id += 1 # The count of the agent_id is always +1 from the one of the last added

    self.pop.append(agent)
  # ---------------------------------

  # ---------------------------------
  def empty(self):
    """
    This function empties the population, without resetting the agent idx counter
    :return:
    """
    self.pop = []
  # ---------------------------------

  # ---------------------------------
  def generate_gen(self):
    """
    This function generates a random genome of size: genome_size
    :return:
    """
    return np.random.normal(0, 1, size=self.genome_size).clip(self.genome_limit[0], self.genome_limit[1])
  # ---------------------------------

  # ---------------------------------
  def save(self, filepath, filename):
    """
    This function saves the population as a pkl file
    :param filepath:
    :param name: Name of the file
    :return:
    """
    try:
      with open(os.path.join(filepath, '{}_{}.pkl'.format(self.name, filename)), 'wb') as file:
        pkl.dump(self.pop, file)
    except Exception as e:
      print('Cannot Save {} {}.'.format(self.name, filename))
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

    if self.params.verbose:
      print('Loading {} from {}'.format(self.name, filepath))
    with open(filepath, 'rb') as file:
      self.pop = pkl.load(file)
    self.agent_id = np.max(self['id'])
  # ---------------------------------