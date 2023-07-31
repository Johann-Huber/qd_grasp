# Created by Giuseppe Paolo 
# Date: 28/07/2020

import numpy as np

# FUnctions here extract the obs from the traj. These obs then are used by the BD extractor

def dummy_obs(traj):
  """
  Get observations from the trajectory coming from the dummy environment
  :param traj:
  :return:
  """
  return np.array([traj[-1][0]]) # Returns the observation of the last element

def walker_2D_obs(traj):
  """
  Get observations from the trajectory coming from the Walker2D environment
  :param traj:
  :return:
  """
  return np.array([t[0] for t in traj]) # t[0] selects the observation part

def collect_ball_obs(traj):
  """
    Get observations from the trajectory coming from the collectball environment
    :param traj:
    :return:
    """
  # In this env the robot position is in the info.
  # Thus we take that and we ignore the traj[0] in which no info is present, given that is the obs from env.reset()
  # Trajectory is normalized.
  # What we return is a list of 2 samples along the traj
  samples = np.concatenate(np.array([t[3]['robot_pos'][:2] for t in traj[1:]])[::int(2000/2)])
  return np.array([samples]) / 600 #TODO VEIRIFCA

def hard_maze_obs(traj):
  """
    Get observations from the trajectory coming from the collectball environment
    :param traj:
    :return:
    """
  # In this env the robot position is in the info.
  # Thus we take that and we ignore the traj[0] in which no info is present, given that is the obs from env.reset()
  # Trajectory is normalized.
  # What we return is a list of 5 samples along the traj
  return np.array([t[3]['robot_pos'][:2] for t in traj[1:]]) / 600

def red_arm_obs(traj):
  """
  Get the observations taken from the traj
  :param traj: list containing gym [obs, reward, done, info]
  :return:
  """
  return np.array([t[3]['End effector pose'] for t in traj[1:]])

def ant_maze_obs(traj):
  """
  Use the x-y position of the robot as observation
  :param traj:
  :return:
  """
  return np.array([(np.array(t[-1]['bc'])-5.)/70. for t in traj[1:]])

def curling_obs(traj):
  """
  Use the xy position of the ball as observations for the bd
  :param traj:
  :return:
  """
  return np.array([t[0][:2] for t in traj]) # t[0] selects the observation part
