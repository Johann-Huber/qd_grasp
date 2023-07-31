# Created by Giuseppe Paolo 
# Date: 28/07/2020

# -----------------------------------------------
# INPUT FORMATTERS
# -----------------------------------------------
def dummy_input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: None
  """
  return None

def walker_input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: None
  """
  return obs

def collect_ball_input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: None
  """
  return obs

def curling_input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: The time
  """
  return obs

def hard_maze_input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: None
  """
  return obs

def red_arm_input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: None
  """
  return obs

def ant_maze_input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return:
  """
  return obs # In the first 2 positions there is the (x,y) pose of the robot

# -----------------------------------------------
# OUTPUT FORMATTERS
# -----------------------------------------------
def output_formatter(action):
  """
  This function formats the output of the controller to extract the action for the env
  :param action:
  :return:
  """
  return action