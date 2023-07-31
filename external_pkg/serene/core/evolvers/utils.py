# Created by Giuseppe Paolo 
# Date: 06/03/2020
import pdb

import numpy as np
import math
from scipy.spatial.distance import squareform, cdist
import matplotlib.pyplot as plt
plt.style.use('seaborn')



def novelty(distances, neighs):
  """
  Calculates the novelty for agent i from the distances given
  :param distances: i row of distance matrix
  :param neighs: number of neighbors used to calculate novelty
  :return: (i, mean_k_dist)
  """
  idx = np.argsort(distances) # Get list of idx from closest to farthest
  mean_k_dist = np.mean(distances[idx[1:neighs + 1]])  # the 1:+1 is necessary cause the position 0 is occupied by the index of the considered element
  return mean_k_dist

def calculate_novelties(bd_set, reference_set, distance_metric='euclidean', novelty_neighs=15, pool=None):
  """
  This function calculates the novelty for each element in the BD set wrt the Reference set
  :param bd_set:
  :param reference_set:
  :param distance_metric: Distance metric with which the novelty is calculated. Default: euclidean
  :param novelty_neighs: Number of neighbors used for novelty calculation. Default: 15
  :param pool: Pool for multiprocessing
  :return:
  """
  try:
    distance_matrix = calculate_distances(bd_set, reference_set, distance_metric=distance_metric)
    if pool is not None:
      novelties = [pool.apply(novelty, args=(distance, novelty_neighs,)) for distance in distance_matrix]
    else:
      novelties = [novelty(distance, novelty_neighs) for distance in distance_matrix]
  except:
    pdb.set_trace()
  return novelties

def calculate_distances(bd_set, reference_set, distance_metric='euclidean'):
  """
  This function is used to calculate the distances between the sets
  :param bd_set:
  :param reference_set:
  :param distance_metric: Distance metric to use. Default: euclidean
  :return:
  """
  if distance_metric == 'euclidean':
    # TODO this operation might become slower when the archive grows. Might have to parallelize as well by doing it myself
    distance_matrix = cdist(bd_set, reference_set, metric='euclidean')
  elif distance_metric == 'mahalanobis':
    distance_matrix = cdist(bd_set, reference_set, metric='mahalanobis')
  elif distance_metric == 'manhattan':
    distance_matrix = cdist(bd_set, reference_set, metric='cityblock')
  elif distance_metric == 'mink_0.1':
    distance_matrix = cdist(bd_set, reference_set, metric='minkowski', p=0.1)
  elif distance_metric == 'mink_1':
    distance_matrix = cdist(bd_set, reference_set, metric='minkowski', p=1)
  elif distance_metric == 'mink_0.01':
    distance_matrix = cdist(bd_set, reference_set, metric='minkowski', p=.01)
  else:
    raise ValueError('Specified distance {} not available.'.format(distance_metric))
  return distance_matrix

def plot_pareto_fronts(points):
  import matplotlib.colors as colors
  cmap = plt.get_cmap('jet')
  plt.figure()
  color = cmap(np.linspace(0, 1, len(points)))

  for idx, front in enumerate(points):
    pp = np.array(points[front])
    plt.scatter(pp[:, 0], pp[:, 1], cmap=cmap, label=front)
    # plt.plot(pp[:, 0], pp[:, 1], '-o', c=color[idx], label=front)
  plt.legend()
  plt.show()

def fast_non_dominated_sort(values1, values2):
  """
  This function sorts the non dominated elements according to the values of the 2 objectives.
  Taken from https://github.com/haris989/NSGA-II
  :param values1: Values of first obj
  :param values2: Values of second obj
  :return: Sorted list of indexes
  """
  S = [[]] * len(values1)
  front = [[]]
  n = [0] * len(values1)
  rank = [0] * len(values1)

  for p in range(len(values1)):
    S[p] = []
    n[p] = 0
    for q in range(len(values1)):
      if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
        if q not in S[p]: S[p].append(q)
      elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
        n[p] += 1
    if n[p] == 0:
      rank[p] = 0
      if p not in front[0]:
        front[0].append(p)

  i = 0
  while front[i] != []:
    Q = []
    for p in front[i]:
      for q in S[p]:
        n[q] -= 1
        if n[q] == 0:
          rank[q] = i + 1
          if q not in Q:
            Q.append(q)
    i += 1
    front.append(Q)
  del front[-1]
  return front

def sort_by_value(front, values):
  """
  This function sorts the front list according to the values
  :param front: List of indexes of elements in the value
  :param values: List of values. Can be longer than the front list
  :return:
  """
  copied_values = values.copy() # Copy so we can modify it
  sorted_list = []
  while len(sorted_list) != len(front):
    min_value = copied_values.index(min(copied_values))
    if min_value in front:
      sorted_list.append(min_value)
    copied_values[min_value] = math.inf
  return sorted_list

def crowding_distance(values1, values2, front):
  """
  This function calculates the crowding distance of the elements in a front
  :param values1:
  :param values2:
  :param front:
  :return:
  """
  distance = [0] * len(front)
  sorted1 = sort_by_value(front, values1)
  sorted2 = sort_by_value(front, values2)
  distance[0] = math.inf
  distance[-1] = math.inf

  for k in range(1, len(front) - 1):
    if max(values1) - min(values1) == 0.:
      distance[k] = np.inf
    else:
      distance[k] = distance[k] + (values1[sorted1[k+1]] - values1[sorted1[k-1]])/(max(values1) - min(values1))
  for k in range(1, len(front) - 1):
    if max(values2) - min(values2) == 0.:
      distance[k] = np.inf
    else:
      distance[k] = distance[k] + (values2[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2) - min(values2))
  return distance















