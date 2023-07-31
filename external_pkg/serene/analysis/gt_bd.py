# Created by Giuseppe Paolo 
# Date: 28/07/2020

import numpy as np

# In this file we specify the functions that from the traj of observations and from the infos extract the ground truth BD.
# On this BD we calculate the coverage and uniformity of the archive.
# But they are not used during the search process. Only at the final evaluation process in the evaluate archive script.

def dummy_gt_bd(traj, max_steps, ts=1):
  """
  This function extract the ground truth BD for the dummy environment
  :param traj:
  :param max_steps: Maximum number of steps the traj can have
  :param ts: percentage of the traj len at which the BD is extracted. In the range [0, 1]. Default: 1
  :return:
  """
  if ts == 1:
    index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
  obs = traj[index][0]
  return obs

def collect_ball_gt_bd(traj, max_steps, ts=1):
  """
	Computes the behavior descriptor from a trajectory.
	A trajectory is a list of tuples (obs,reward,end,info)  (depends on the environment, see gym_env.py).
	For the maze, we output the last robot position (x,y only, we discard theta).
	:param traj:
  :param max_steps: Maximum number of steps the traj can have
  :param ts: percentage of the traj len at which the BD is extracted. In the range [0, 1]. Default: 1
  :return:
	"""
  if ts == 1:
    index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
  obs = np.array(traj[index][-1]['robot_pos'][:2])/600.
  return obs

def hard_maze_gt_bd(traj, max_steps, ts=1):
  """
	Computes the behavior descriptor from a trajectory.
	A trajectory is a list of tuples (obs,reward,end,info)  (depends on the environment, see gym_env.py).
	For the maze, we output the last robot position (x,y only, we discard theta).
	:param traj:
  :param max_steps: Maximum number of steps the traj can have
  :param ts: percentage of the traj len at which the BD is extracted. In the range [0, 1]. Default: 1
  :return:
	"""
  if ts == 1:
    index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
  obs = np.array(traj[index][-1]['robot_pos'][:2])/600
  return obs

def red_arm_gt_bd(traj, max_steps, ts=1):
  """
  	Computes the behavior descriptor from a trajectory.
  	:param traj: gym env list of observations
  	:param info: gym env list of info
    :param max_steps: Maximum number of steps the traj can have
    :param ts: percentage of the traj len at which the BD is extracted. In the range [0, 1]. Default: 1
    :return:
  	"""
  if ts == 1:
    index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
  obs = traj[index][-1]['End effector pose']
  return obs

def ant_maze_gt_bd(traj, max_steps, ts=1):
  if ts == 1:
    index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
  obs = traj[index][0][:2]
  return obs

def curling_gt_bd(traj, max_steps, ts=1):
  if ts == 1:
    index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
  obs = traj[index][0][:2]
  return obs