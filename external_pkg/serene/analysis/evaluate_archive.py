# Created by Giuseppe Paolo 
# Date: 28/07/2020

import os
import parameters
import multiprocessing as mp
from core import Evaluator
from analysis import utils
from external_pkg.serene.environments.environments import registered_envs
import pickle as pkl
import argparse
import gc
from external_pkg.serene.core.behavior_descriptors.behavior_descriptors import BehaviorDescriptor
from external_pkg.serene.analysis.gt_bd import *
from copy import deepcopy

evaluator = None
main_pool = None

class EvalArchive(object):
  """
  This function is used to evaluate the archive of an experiment
  """
  def __init__(self, exp_path, agents=None):
    """
    Constructor
    """
    self.params = parameters.Params()
    self.params.load(os.path.join(exp_path, '_params.json'))
    self.exp_path = exp_path
    self.agents = agents
    self.bd_extractor = BehaviorDescriptor(self.params)
    self.pca = None

    self.traj_to_obs = registered_envs[self.params.env_name]['traj_to_obs']
    self.grid_params = registered_envs[self.params.env_name]['grid']

    if not os.path.exists(os.path.join(self.exp_path, 'analyzed_data')):
      os.mkdir(os.path.join(self.exp_path, 'analyzed_data'))

    # GT_BD is the ground truth bd that is used to calculate the CVG
    if self.params.env_name == 'Dummy':
      self.gt_bd_extractor = dummy_gt_bd
      self.goals = 0
    elif self.params.env_name == 'Walker2D':
      self.gt_bd_extractor = dummy_gt_bd
      self.goals = 0
    elif self.params.env_name == 'CollectBall':
      self.gt_bd_extractor = collect_ball_gt_bd
      self.goals = 0
    elif self.params.env_name == 'NDofArm':
      self.gt_bd_extractor = red_arm_gt_bd
      self.goals = 3
    elif self.params.env_name == 'AntMaze':
      self.gt_bd_extractor = ant_maze_gt_bd
      self.goals = 2
    elif self.params.env_name == 'HardMaze':
      self.gt_bd_extractor = hard_maze_gt_bd
      self.goals = 2
    elif self.params.env_name == 'Curling':
      self.gt_bd_extractor = curling_gt_bd
      self.goals = 0
    else:
      print("No GT BD given for: {}".format(self.params.env_name))
      raise ValueError

  def calculate_cvg_unif_me(self):
    """
    Calculates CVG and UNIF for methods with only archive
    :return:
    """
    cvg = []
    unif = []
    evaluations = []

    # Load archives
    archive = utils.load_data(self.exp_path,
                              generation=-1,
                              params=self.params,
                              container_type='archive',
                              info=['gt_bd']
                              )
    archive = archive[-1]
    count = 0
    for filled in archive['filled']:
      count += 1
      cvg.append(count/self.grid_params['bins'] ** 2)
      unif.append(1)
      evaluations.append(filled)

    # Interpolate arrays
    points = np.linspace(0, self.params.evaluation_budget, 1000)
    cvg = np.interp(points, evaluations, cvg)
    unif = np.interp(points, evaluations, unif)
    evaluations = np.interp(points, evaluations, evaluations)

    data = {'cvg': cvg, 'unif': unif, 'eval': evaluations}
    with open(os.path.join(self.exp_path, 'analyzed_data', 'cvg_unif_by_eval.pkl'), 'wb') as f:
      pkl.dump(data, f)

  def calculate_cvg_unif_ns(self):
    """
    Calculates CVG and UNIF for methods with archive and pop separated
    This function goes through the offs of all the generations to gather CVG and UNIF data.
    For the first generation, it starts also with the pop.
    :return:
    """
    grid = utils.CvgGrid(self.params, self.grid_params)
    cvg = []
    unif = []
    evaluations = []

    # Load first init population
    init_pop = utils.load_data(self.exp_path,
                               generation=1,
                               params=self.params,
                               container_type='population',
                               info=['evaluated', 'gt_bd']
                               )
    # Store them in the CVG grid
    for eval, bd in zip(init_pop[1]['evaluated'], init_pop[1]['gt_bd']):
      grid.store({'evaluated': eval, 'gt_bd': bd})
      cvg.append(utils.calculate_coverage(grid.grid))
      unif.append(utils.calculate_uniformity(grid.grid))
      evaluations.append(eval)

    # Now load all the offsprings
    offsprings = utils.load_data(self.exp_path,
                                 generation=None,
                                 params=self.params,
                                 container_type='offsprings',
                                 info=['evaluated', 'gt_bd'])

    # Now store all the offs in the grid and calculate cvg and unif
    gen = 1
    while gen in offsprings:
      if gen % 100 == 0: print("Working on gen: {}".format(gen))
      for eval, bd in zip(offsprings[gen]['evaluated'], offsprings[gen]['gt_bd']):
        grid.store({'evaluated': eval, 'gt_bd': bd})
        cvg.append(utils.calculate_coverage(grid.grid))
        unif.append(utils.calculate_uniformity(grid.grid))
        evaluations.append(eval)
      gen += 1

    # Subsample so to have only 1k points
    points = np.linspace(0, self.params.evaluation_budget, 1000)
    cvg = np.interp(points, evaluations, cvg)
    unif = np.interp(points, evaluations, unif)
    evaluations = np.interp(points, evaluations, evaluations)

    data = {'cvg': cvg, 'unif': unif, 'eval': evaluations}
    with open(os.path.join(self.exp_path, 'analyzed_data', 'cvg_unif_by_eval.pkl'), 'wb') as f:
      pkl.dump(data, f)

  def calculate_cvg_unif(self):
    """
    Selects between the 2 methods
    :return:
    """
    if self.params.exp_type in ['ME', 'CMA-ME']:
      self.calculate_cvg_unif_me()
    else:
      self.calculate_cvg_unif_ns()

  def rew_by_eval(self):
    """
    This function extracts the performances at each evaluation step
    :return:
    """
    # All the informations we need are in the final archives
    info = ['reward', 'evaluated', 'rew_area']
    archive = utils.load_data(self.exp_path, generation=-1, params=self.params, container_type='archive', info=info)
    data = archive[-1]
    del archive

    if os.path.exists(os.path.join(self.exp_path, 'rew_archive_final.pkl')):
      rew_archive = utils.load_data(self.exp_path, generation=-1, params=self.params, container_type='rew_archive', info=info)
      rew_archive = rew_archive[-1]
      # Merge archive and rew_arch data
      for i in info:
        data[i] = data[i] + rew_archive[i]
      del rew_archive

    # Sort data according to evaluation step
    eval = deepcopy(data['evaluated'])
    data = {k: np.array([x for _, x in sorted(zip(eval, v), key=lambda x: x[0])]) for k, v in data.items()}
    data['rew_area'] = np.stack(data['rew_area'])

    # Init the array with zero reward at eval zero
    rewards = {"rew_area_{}".format(i+1): [np.array([0, 0])] for i in range(self.goals)}

    for eval, rew, area in zip(data['evaluated'], data['reward'], data['rew_area']):
      if area is not None:
        rewards["rew_area_{}".format(area)].append(np.array([eval, rew]))

    interp_rew = {}
    points = np.linspace(0, self.params.evaluation_budget, 1000)
    for k in rewards:
      # Interpolates reward in steps
      rew = np.stack(rewards[k]) # [[evaluation, reward]]
      rew[:, 0] = rew[:, 0]*1000/self.params.evaluation_budget
      rew[:, 1] = np.maximum.accumulate(rew[:, 1])
      max_rew = np.zeros_like(points)
      for idx in range(len(rew[:, 0])):
        e, r = rew[idx]
        bottom = int(rew[idx, 0])
        # Calculate upper limit
        if idx < len(rew[:, 0])-1:
          upper = int(rew[idx+1, 0])
        else:
          upper = len(max_rew)
        # Update max_rew array
        max_rew[bottom:upper] = r
      interp_rew[k] = max_rew

    interp_rew['eval'] = points

    with open(os.path.join(self.exp_path, 'analyzed_data', 'rew_by_eval.pkl'), 'wb') as f:
      pkl.dump(interp_rew, f)

  def final_arch_gt_bd(self):
    """
    This function returns the gt_bd of only the final archive
    :return:
    """
    info = ['gt_bd']
    # EXPLORATION ARCHIVE
    archive_gt_bd = utils.load_data(self.exp_path,
                                    info=info + ['reward'],
                                    generation=-1,
                                    params=self.params,
                                    container_type='archive',
                                    )

    # REWARD ARCHIVE
    # Check if rew_archive exists
    filename = 'rew_archive_final.pkl'
    if os.path.exists(os.path.join(self.exp_path, filename)):
      rew_archive_gt_bd = utils.load_data(self.exp_path,
                                          info=info,
                                          generation=-1,
                                          params=self.params,
                                          container_type='rew_archive',
                                          )
    else:
      info = ['reward', 'gt_bd']
      rew_archive_gt_bd = utils.load_data(self.exp_path,
                                          info=info,
                                          generation=-1,
                                          params=self.params,
                                          container_type='archive',
                                          )
      for gen in rew_archive_gt_bd:
        rew_archive_gt_bd[gen] = {
          'gt_bd': [rew_archive_gt_bd[gen]['gt_bd'][idx] for idx in range(len(rew_archive_gt_bd[gen]['gt_bd'])) if
                    rew_archive_gt_bd[gen]['reward'][idx] > 0]}

    archive_gt_bd = np.stack(archive_gt_bd[-1]['gt_bd'])
    rew_archive_gt_bd = rew_archive_gt_bd[-1]['gt_bd']
    try:
      rew_archive_gt_bd = np.stack(rew_archive_gt_bd)
    except:
      pass
    # Save everything
    gt_bd = {'archive': archive_gt_bd,
             'rew archive': rew_archive_gt_bd}
    with open(os.path.join(self.exp_path, 'analyzed_data', 'final_gt_bd.pkl'), 'wb') as f:
      pkl.dump(gt_bd, f)

  def save_gt_bd(self):
    """
    This function calculates the coverage and uniformity used for plotting.
    :param trajectories: of all the agents in the run
    :param generation: Generation from which the trajectories are from
    :return:
    """
    info = ['gt_bd']
    # POPULATION
    population_gt_bd = utils.load_data(self.exp_path,
                                       info=info,
                                       generation=None,
                                       params=self.params,
                                       container_type='population',
                                       )
    max_gen = max(list(population_gt_bd.keys())) # Find max_gen
    population_gt_bd['final'] = population_gt_bd[max_gen]  # Save final like this as well, so it's easier to recover

    # OFFSPRINGS
    offsprings_gt_bd = utils.load_data(self.exp_path,
                                       info=info,
                                       generation=None,
                                       params=self.params,
                                       container_type='offsprings',
                                       )
    offsprings_gt_bd['final'] = offsprings_gt_bd[max_gen]  # Save final like this as well, so it's easier to recover

    # EXPLORATION ARCHIVE
    archive_gt_bd = utils.load_data(self.exp_path,
                                    info=info + ['reward'],
                                    generation=None,
                                    params=self.params,
                                    container_type='archive',
                                   )
    archive_gt_bd['final'] = archive_gt_bd[max_gen] # Save final like this as well, so it's easier to recover

    # REWARD ARCHIVE
    # Check if rew_archive exists
    filename = 'rew_archive_final.pkl'
    if os.path.exists(os.path.join(self.exp_path, filename)):
      rew_archive_gt_bd = utils.load_data(self.exp_path,
                                    info=info,
                                    generation=None,
                                    params=self.params,
                                    container_type='rew_archive',
                                   )
    else:
      info = ['reward', 'gt_bd']
      rew_archive_gt_bd = utils.load_data(self.exp_path,
                                          info=info,
                                          generation=None,
                                          params=self.params,
                                          container_type='archive',
                                          )
      for gen in rew_archive_gt_bd:
        rew_archive_gt_bd[gen] = {'gt_bd': [rew_archive_gt_bd[gen]['gt_bd'][idx] for idx in range(len(rew_archive_gt_bd[gen]['gt_bd'])) if rew_archive_gt_bd[gen]['reward'][idx] > 0]}
    try:
      max_gen = max(list(rew_archive_gt_bd.keys()))  # Find max_gen
      rew_archive_gt_bd['final'] = rew_archive_gt_bd[max_gen]  # Save final like this as well, so it's easier to recover
    except:
      print("No reward found in run: {}".format(self.exp_path))
      rew_archive_gt_bd['final'] = {'gt_bd': []}

    # Given that load_data returns a dict {gen:{info: [data]}} we remove the depth of the info cause we already know it's gt_bd
    population_gt_bd = {gen: population_gt_bd[gen]['gt_bd'] for gen in population_gt_bd}
    offsprings_gt_bd = {gen: offsprings_gt_bd[gen]['gt_bd'] for gen in offsprings_gt_bd}
    archive_gt_bd = {gen: archive_gt_bd[gen]['gt_bd'] for gen in archive_gt_bd}
    rew_archive_gt_bd = {gen: rew_archive_gt_bd[gen]['gt_bd'] for gen in rew_archive_gt_bd}

    # Save everything
    gt_bd = {'population': population_gt_bd,
             'offsprings': offsprings_gt_bd,
             'archive': archive_gt_bd,
             'rew archive': rew_archive_gt_bd}
    with open(os.path.join(self.exp_path, 'analyzed_data', 'gt_bd.pkl'), 'wb') as f:
      pkl.dump(gt_bd, f)

def main(path, args):
  arch_eval = EvalArchive(path, agents=args.agents)

  print()
  print("Calculating cvg and unif by evaluation step")
  # ---------------------
  arch_eval.calculate_cvg_unif()
  gc.collect()
  # ---------------------

  print()
  print("Calculating reward by evaluation step")
  # ---------------------
  arch_eval.rew_by_eval()
  gc.collect()
  # ---------------------

  print()
  print('Extracting final GT BDs')
  arch_eval.final_arch_gt_bd()
  gc.collect()

  if args.abd:
    print()
    print("Extracting complete GT BDs")
    # ---------------------
    arch_eval.save_gt_bd()
    gc.collect()
    # ---------------------

  del arch_eval
  gc.collect()

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Run archive eval script')
  parser.add_argument('-p', '--path', help='Path of experiment')
  parser.add_argument('-mp', '--multiprocessing', help='Multiprocessing', action='store_true')
  parser.add_argument('-g', '--generation', help='Generation for which to evaluate the archive', type=int, default=None)
  parser.add_argument('-a', '--agents', help='Agents to evaluate', type=int, default=None)
  parser.add_argument('--multi', help='Flag to give in case multiple runs have to be evaluated', action='store_true')
  parser.add_argument('-abd', help="Generate GT BD for all generations", action='store_true')

  args = parser.parse_args()
  #["-p", '/home/giuseppe/src/cmans/experiment_data/Curling_ME_std', '-g', '-1', '--multi'])

  if not args.multi:
    paths = [args.path]
  else:
    if not os.path.exists(args.path):
      raise ValueError('Path does not exist: {}'.format(args.path))
    raw_paths = [x for x in os.walk(args.path)]
    raw_paths = raw_paths[0]
    paths = [os.path.join(raw_paths[0], p) for p in raw_paths[1]]

  if args.multiprocessing:
    pool = mp.Pool()
    pool.starmap(main, zip(paths, [args]*len(paths)))
  else:
    for path in paths:
      print('Working on: {}'.format(path))
      main(path, args)