# Created by Giuseppe Paolo 
# Date: 28/07/2020

import numpy as np
import os
from scipy.spatial.distance import jensenshannon
from external_pkg.serene.core.population import Archive, Population, Grid
import re

class CvgGrid(Grid):
  """
  This class implements the grid used to calculate the CVG and UNIF
  """
  def init_grid(self):
    return np.zeros([self.grid_params['bins']] * self.bd_dimensions)

  def store(self, agent):
    """
    Store data in the archive as a list of: (genome, gt_bd, bd, traj).
    No need to store the ID given that we store the genome.
    Saving as a tuple instead of a dict makes the append operation faster

    It also checks if the grid cell is already occupied. In case it is, saves the one with the highest fitness

    :param agent: agent to store
    :return:
    """
    assert len(agent['gt_bd']) == self.bd_dimensions, print('GT_BD of wrong size. Given: {} - Expected: {}'.format(len(agent['gt_bd']), self.bd_dimensions))
    cell = self._find_cell(agent['gt_bd'])

    if self.grid[cell] is None:
      # Init cell and add the evaluated to the filled tracker
      self.grid[cell] = 1
      self.filled_tracker.append(agent['evaluated'])
    else:
      # Increase cell count by one
      self.grid[cell] += 1


def calculate_coverage(occupied_grid):
  """
  This function calculated the coverage percentage from the grid
  :param occupied_grid
  :return:
  """
  coverage = np.count_nonzero(occupied_grid)/occupied_grid.size
  return coverage

def calculate_uniformity(grid):
  """
  This function calculates the uniformity of the normed grid, that is the histogram
  :param normed_grid
  :return:
  """
  normed_grid = grid/np.sum(grid)
  uniform_grid = np.ones_like(normed_grid)/normed_grid.size
  return 1-jensenshannon(normed_grid.flatten(), uniform_grid.flatten())

def get_grid(points, grid_parameters, kde=True):
  """
  This function calculates the normed histogram and the grid of occupied cells.
  :param points:
  :param grid_parameters
  :return: histogram, occupied_grid
  """
  hist, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1],
                                          bins=[grid_parameters['bins'], grid_parameters['bins']],
                                          range=[[grid_parameters['min_coord'][0], grid_parameters['max_coord'][0]],
                                                 [grid_parameters['min_coord'][1], grid_parameters['max_coord'][1]]],
                                          density=False)
  hist = hist.T[::-1, :]
  occupied_grid = hist.copy()
  occupied_grid[occupied_grid > 0] = 1

  if kde:
    from scipy.stats.kde import gaussian_kde
    k = gaussian_kde(np.vstack([points[:, 0], points[:, 1]]))
    xi, yi = np.mgrid[grid_parameters['min_coord'][0]:grid_parameters['max_coord'][0]:points[:, 0].size ** 0.5 * 1j,
                      grid_parameters['min_coord'][1]:grid_parameters['max_coord'][1]:points[:, 1].size ** 0.5 * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
  else:
    xi = None
    yi = None
    zi = None
  return hist, occupied_grid, [xi, yi, zi]

def get_runs_list(path):
  """
  This function returns the list of run folders in the path, by verifying that they are in the right format
  :param path:
  :return:
  """
  assert os.path.exists(path), "The path {} does not exists!".format(path)
  r = re.compile(".{4}_.{2}_.{2}_.{2}:.{2}_.{6}")
  runs = [run for run in os.listdir(path) if r.match(run)]
  return runs

def load_data(folder, info=None, generation=None, params=None, container_type="archive"):
  """
  This function loads all the archives, by generation in an experiment folder.
  :param folder: The folder of the experiment
  :param info: List of information to load from the archive. If None the whole archive is loaded
  :param generation: generation for which to load the archive. If None, load all the archives. If -1 loads the final
  :param params: run parameters to load the datastructure
  :param container_type: One among: [population, offsprings, archive, rew_archive]
  :returns dict in the shape: {generation:{label: data}}
  """
  if container_type == 'archive' or container_type == 'rew_archive':
    DataStruct = Archive
  elif container_type == 'population' or container_type == 'offsprings':
    DataStruct = Population
  else:
    raise ValueError('Container types available: {} - Given: {}'.format('[population, offsprings, archive, rew_archive]',
                                                                        container_type))
  data = {}
  # Load all the archives
  if generation == None:
    # Get all archive names
    r = re.compile('{}_gen_.*.pkl'.format(container_type))
    files = [file for file in os.listdir(folder) if r.match(file) is not None]

    generation = 0
    while len(files) > 0:
      generation += 1
      filename = "{}_gen_{}.pkl".format(container_type, generation)

      # If the file does not exists, skip the loading
      if not os.path.exists(os.path.join(folder, filename)):
        continue

      # If it exists load it and remove its name from the list
      data_struct = DataStruct(params)
      files.remove(filename)
      data_struct.load(os.path.join(folder, filename))
      if info is None:
        data[generation] = data_struct  # Load whole archive
      else:
        data[generation] = {}
        if hasattr(data_struct, 'filled_tracker'):
          data[generation]['filled'] = data_struct.filled_tracker
        for label in info:  # Save only the needed info
          data[generation][label] = data_struct[label]  # Each info is saved as a list of values
  else:
    if generation == -1: name = "{}_final.pkl".format(container_type)
    else: name = "{}_gen_{}.pkl".format(container_type, generation)

    data_struct = DataStruct(params)
    data_struct.load(os.path.join(folder, name))
    if info is None:
      data[generation] = data_struct  # Load whole archive
    else:
      data[generation] = {}
      if hasattr(data_struct, 'filled_tracker'):
        data[generation]['filled'] = data_struct.filled_tracker
      for label in info:  # Save only the needed info
        data[generation][label] = data_struct[label]  # Each info is saved as a list of values

  return data