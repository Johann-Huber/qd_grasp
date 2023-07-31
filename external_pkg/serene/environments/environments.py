# Created by Giuseppe Paolo 
# Date: 27/07/2020

from external_pkg.serene.core.controllers import *
from external_pkg.serene.environments.io_formatters import *
from external_pkg.serene.core.behavior_descriptors.trajectory_to_observations import *
from external_pkg.serene.analysis.gt_bd import *

try: import gym_dummy
except: print("Gym dummy not installed")

try: import gym_collectball
except: print("Gym collect ball not installed")

try: import gym_redarm
except: print("Gym redundant arm not installed")

try: import ant_maze
except: print("Ant maze not installed")

try: import gym_fastsim
except: print('Gym fastsim not installed')

try: import gym_billiard
except: print('Gym billiard not installed')

registered_envs = {}

registered_envs['NAME'] = {
  'gym_name': None,
  'controller': None,
  'input_formatter': None,
  'output_formatter': None,
  "traj_to_obs": None,
}


registered_envs['Dummy'] = {
  'gym_name': 'Dummy-v0',
  'controller': {
    'controller': DummyController,
    'input_formatter': dummy_input_formatter,
    'output_formatter': output_formatter,
    'input_size': 2,
    'output_size':2,
    'name': 'dummy',
  },
  'traj_to_obs': dummy_obs,
  'gt_bd': dummy_gt_bd,
  'max_steps': 1,
  'grid':{
    'min_coord':[-1,-1],
    'max_coord':[1, 1],
    'bins':50
  }
}

registered_envs['Walker2D'] = {
  'gym_name': 'Walker2D-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': walker_input_formatter,
    'output_formatter': output_formatter,
    'input_size': 2,
    'output_size': 2,
    'hidden_layers': 2,
    'name': 'dummy',
  },
  'traj_to_obs': walker_2D_obs,
  'gt_bd': dummy_gt_bd,
  'max_steps': 50,
  'grid':{
    'min_coord':[-1,-1],
    'max_coord':[1, 1],
    'bins':50
  }
}

registered_envs['CollectBall'] = {
  'gym_name': 'CollectBall-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': collect_ball_input_formatter,
    'output_formatter': output_formatter,
    'input_size': 10,
    'output_size': 3,
    'hidden_layers': 2,
    'name': 'neural'
  },
  'traj_to_obs': collect_ball_obs,
  'gt_bd': collect_ball_gt_bd,
  'max_steps': 2000,
  'grid':{
    'min_coord':[0, 0]*2,
    'max_coord':[1, 1]*2, # We use observations that are already scaled
    'bins':50
  }
}

registered_envs['HardMaze'] = {
  'gym_name': 'FastsimSimpleNavigation-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': hard_maze_input_formatter,
    'output_formatter': output_formatter,
    'input_size': 5,
    'output_size': 2,
    'hidden_layers': 2,
    'name': 'neural'
  },
  'traj_to_obs': hard_maze_obs,
  'gt_bd': hard_maze_gt_bd,
  'max_steps': 2000,
  'grid':{
    'min_coord':[0, 0],
    'max_coord':[1, 1], # We use observations that are already scaled
    'bins':50
  }
}

registered_envs['NDofArm'] = {
  'gym_name': 'RedundantArm-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': red_arm_input_formatter,
    'output_formatter': output_formatter,
    'input_size': 20,
    'output_size': 20,
    'hidden_layers': 2,
    'name': 'neural'
  },
  'traj_to_obs': red_arm_obs,
  'gt_bd': red_arm_gt_bd,
  'max_steps': 100,
  'grid': {
    'min_coord':[0, 0],
    'max_coord':[1, 1],
    'bins':50
  }
}

registered_envs['AntMaze'] = {
  'gym_name': 'AntObstacles-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': ant_maze_input_formatter,
    'output_formatter': output_formatter,
    'input_size': 29,
    'output_size': 8,
    'hidden_layers': 3,
    'hidden_layer_size': 10,
    'name': 'neural'
  },
  'traj_to_obs': ant_maze_obs,
  'gt_bd': ant_maze_gt_bd,
  'max_steps': 3000,
  'grid': {
    'min_coord':[-.5, -.5],
    'max_coord':[.5, .5],
    'bins':50
  }
}

registered_envs['Curling'] = {
  'gym_name': 'Curling-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': curling_input_formatter,
    'output_formatter': output_formatter,
    'input_size': 6,
    'output_size':2,
    'hidden_layers': 3,
    'name': 'neural',
  },
  'traj_to_obs': curling_obs,
  'gt_bd': curling_gt_bd,
  'max_steps': 500,
  'grid':{
    'min_coord':[-1.35,-1.35],
    'max_coord':[1.35, 1.35],
    'bins':50
  }
}