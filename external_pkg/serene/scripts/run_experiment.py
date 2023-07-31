# Created by Giuseppe Paolo 
# Date: 28/07/2020

import sys, os
import setuptools
from parameters import Params
import numpy as np
import random
import traceback
from progress.bar import Bar
import argparse
import multiprocessing as mp
from parameters import params
from external_pkg.serene.core.searcher import Searcher
import json
from external_pkg.serene.analysis.logger import Logger

import datetime
from external_pkg.serene.environments.environments import registered_envs

experiments = ['NS',
               'CMA-ES',
               'CMA-NS',
               'NSGA-II',
               'SERENE',
               'ME',
               'CMA-ME',
               'RND']

if __name__ == "__main__":
  # To check why these options are here:
  # 1. https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
  # 2. https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
  mp.set_start_method('spawn')
  #  mp.set_sharing_strategy('file_system') # Fundamental otherwise crashes complaining that too many files are open

  parser = argparse.ArgumentParser('Run evolutionary script')
  parser.add_argument('-env', '--environment', help='Environment to use', choices=list(registered_envs.keys()))
  parser.add_argument('-exp', '--experiment', help='Experiment type. Defines the behavior descriptor', choices=experiments),
  parser.add_argument('-ver', '--version', help='Version of the experiment', default='std')
  parser.add_argument('-sp', '--save_path', help='Path where to save the experiment')
  parser.add_argument('-mp', '--multiprocesses', help='How many parallel workers need to use', type=int)
  parser.add_argument('-p', '--pop_size', help='Size of the population', type=int)
  parser.add_argument('-eb', '--evaluation_budget', help='Number of evaluations to perform', type=int)
  parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
  parser.add_argument('-li', help='Local improvement', action='store_true')
  parser.add_argument('--restart_gen', help='Generation at which to restart. It will load from the savepath', type=int)

  args = parser.parse_args()

  params.version = args.version
  if args.environment is not None: params.env_name = args.environment
  if args.experiment is not None: params.exp_type = args.experiment
  if args.save_path is not None: params.save_path = os.path.join(args.save_path, params.save_dir)
  if args.multiprocesses is not None: params.multiprocesses = args.multiprocesses
  if args.pop_size is not None: params.pop_size = args.pop_size
  if args.evaluation_budget is not None: params.evaluation_budget = args.evaluation_budget
  if args.verbose is True: params.verbose = args.verbose
  if args.li is True: params.local_improvement = False

  print("SAVE PATH: {}".format(params.save_path))
  params.save()

  if params.seed is not None:
    np.random.seed(params.seed)

  bar = Bar('Evals:', max=params.evaluation_budget, suffix='[%(index)d/%(max)d] - Avg time per eval: %(avg).3fs - Elapsed: %(elapsed_td)s')

  searcher = Searcher(params)

  if args.restart_gen is not None:
    print("Restarting:")
    print("\t Restarting from generation {}".format(args.restart_gen))
    print("\t Loading from: {}".format(args.save_path))
    searcher.load_generation(args.restart_gen, args.save_path)
    print("\t Loading done.")

  gen_times = []
  evaluated_points = 0
  previous_count = evaluated_points
  while evaluated_points < params.evaluation_budget:
    if params.verbose:
      print("Generation: {}".format(searcher.generation))

    try:
      gen_time, evaluated_points = searcher.chunk_step()
      if evaluated_points % 100000 == 0:
        print("Evaluated: {}".format(evaluated_points))
      gen_times.append(gen_time)

    except KeyboardInterrupt:
      print('User interruption. Saving.')
      searcher.population.save(params.save_path, 'gen_{}'.format(searcher.generation))
      searcher.evolver.archive.save(params.save_path, 'gen_{}'.format(searcher.generation))
      searcher.offsprings.save(params.save_path, 'gen_{}'.format(searcher.generation))
      if hasattr(searcher.evolver, 'rew_archive'):
        searcher.evolver.rew_archive.save(params.save_path, 'gen_{}'.format(searcher.generation))
      bar.finish()
      total_time = np.sum(gen_times)
      Logger.data['Time'] = str(datetime.timedelta(seconds=total_time))
      Logger.data['Evaluated points'] = searcher.evolver.evaluated_points
      Logger.data['Generations'] = searcher.generation + 1
      Logger.data['End'] = 'User interrupt'
      searcher.close()
      with open(os.path.join(params.save_path, 'recap.json'), 'w') as fp:
        json.dump(Logger.data, fp)
      break

    except Exception as e:
      print('Exception occurred.')
      print(traceback.print_exc())
      total_time = np.sum(gen_times)
      Logger.data['Time'] = str(datetime.timedelta(seconds=total_time))
      Logger.data['Evaluated points'] = searcher.evolver.evaluated_points
      Logger.data['Generations'] = searcher.generation + 1
      Logger.data['End'] = traceback.print_exc()
      with open(os.path.join(params.save_path, 'recap.json'), 'w') as fp:
        json.dump(Logger.data, fp)
      searcher.close()
      bar.finish()
      sys.exit()

    bar.next(n=evaluated_points - previous_count)
    previous_count = evaluated_points

  searcher.population.save(params.save_path, 'final')
  searcher.evolver.archive.save(params.save_path, 'final')
  searcher.offsprings.save(params.save_path, 'final')
  if hasattr(searcher.evolver, 'rew_archive'):
    searcher.evolver.rew_archive.save(params.save_path, 'final')

  total_time = np.sum(gen_times)
  print("Total time: {}".format(str(datetime.timedelta(seconds=total_time))))
  Logger.data['Time'] = str(datetime.timedelta(seconds=total_time))
  Logger.data['Evaluated points'] = evaluated_points
  Logger.data['Generations'] = searcher.generation + 1
  Logger.data['End'] = 'Finished'
  Logger.data['Emitters'] = searcher.evolver.emitters_data

  with open(os.path.join(params.save_path, 'recap.json'), 'w') as fp:
    json.dump(Logger.data, fp)

  searcher.close()
  print('Done.')














































