# Created by Giuseppe Paolo 
# Date: 28/07/2020

import numpy as np
from external_pkg.serene.environments.environments import registered_envs

class BehaviorDescriptor(object):
  """
  This class defines the behavior descriptor.
  Each BD function returns a behavior descriptor and the surprise associated with it.
  (Have to do it here cause of efficiency reasons, otherwise will have to rerun the images through the autoencoder later)
  To add more BD add the function and the experiment type in the init.
  """
  def __init__(self, parameters):
    self.params = parameters
    self.traj_max_len = registered_envs[self.params.env_name]['max_steps']

    if self.params.exp_type == 'NS':
      self.descriptor = self.novelty_bd
    elif self.params.exp_type == 'CMA-ES':
      self.descriptor = self.dummy_bd
    elif self.params.exp_type == 'CMA-NS':
      self.descriptor = self.novelty_bd
    elif self.params.exp_type == 'NSGA-II':
      self.descriptor = self.novelty_bd
    elif self.params.exp_type == 'SERENE':
      self.descriptor = self.novelty_bd
    elif self.params.exp_type == 'ME':
      self.descriptor = self.novelty_bd
    elif self.params.exp_type == 'CMA-ME':
      self.descriptor = self.novelty_bd
    elif self.params.exp_type == 'RND':
      self.descriptor = self.dummy_bd
    else:
      raise ValueError("No behavior descriptor defined for experiment type {}".format(self.params.exp_type))

    # Function to extract the observations from the complete trajectory
    self.traj_to_obs = registered_envs[self.params.env_name]['traj_to_obs']

  def __call__(self, traj, agent, **kwargs):
    return self.descriptor(traj, agent, **kwargs)

  def novelty_bd(self, traj, agent):
    """
    This function returns the last observation extracted from the trajectory
    :param traj: complete trajectory consisting in a list of [obs, rew, done, info]
    :param agent:
    :return: ground truth BD
    """
    return self.traj_to_obs(traj)[-1]

  def dummy_bd(self, traj, agent):
    """
    This function implements a dummy bd for algorithms that do not use it
    :param agent:
    :return: 0
    """
    return 0