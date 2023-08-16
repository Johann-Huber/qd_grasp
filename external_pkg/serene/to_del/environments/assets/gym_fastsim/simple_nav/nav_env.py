import os, subprocess, time, signal
import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import math
import random
import logging
from skimage import draw
import netpbmfile


import pyfastsim as fs

logger = logging.getLogger(__name__)

default_env = "assets/LS_maze_hard.xml"
#default_env = "assets/example.xml"

sticky_walls = False # default is True in libfastsim... but false in the fastsim sferes2 module -> stick (haha) with that


def sqdist(x,y):
	return (x[0]-y[0])**2+(x[1]-y[1])**2

def dist(x,y):
	return math.sqrt(sqdist(x,y))

def reward_distance_multigoal(navenv):
	"""
	Multiple goal. Once in the goal area, the closer the better. Only works on the final position
	:param navenv:
	:return:
	"""
	if navenv.step_count >= navenv.max_steps:
		curpos = navenv.current_pos
		for goal_idx, goal in enumerate(navenv.goalPos):
			if (dist(curpos, goal) <= navenv.goalRadius[goal_idx]) and (navenv.reward_area is None or navenv.reward_area == goal_idx):
				navenv.reward_area = goal_idx
				return (navenv.goalRadius[goal_idx] - dist(curpos, goal)) / navenv.goalRadius[goal_idx]
	return 0

def reward_binary_multigoal(navenv):
	"""
	Deals with multiple goals.
	:param navenv:
	:return:
	"""
	curpos = navenv.current_pos
	for goal_idx, goal in enumerate(navenv.goalPos):
		if (dist(curpos, goal) <= navenv.goalRadius[goal_idx]) and (navenv.reward_area is None or navenv.reward_area == goal_idx):
			navenv.reward_area = goal_idx
			return 1.
	return 0.

def reward_binary_goal_based(navenv):
	""" Reward of 1 is given when close enough to the goal. """
	curpos = navenv.current_pos
	if (dist(curpos,navenv.goalPos[0])<=navenv.goalRadius[0]):
		return 1.
	else:
		return 0.

def reward_minus_energy(navenv):
	""" Reward = minus sum of absolute values of motor orders"""
	r = -(np.abs(navenv.v1_motor_order) + np.abs(navenv.v2_motor_order))
	return r

def reward_displacement(navenv):
	""" Reward = distance to previous position"""
	r = dist(navenv.current_pos, navenv.old_pos)
	return r

def no_reward(navenv):
	""" No reward"""
	return 0.


reward_functions = { "binary_goalbased":reward_binary_goal_based,
				"minimize_energy":reward_minus_energy,
				"displacement":reward_displacement,
				"binary_multigoal":reward_binary_multigoal,
				"distance_multigoal": reward_distance_multigoal,
				"none":no_reward,
				None:no_reward}



class SimpleNavEnv(gym.Env):
	def __init__(self,xml_env, reward_func="distance_multigoal", display=False, light_sensor_range=200., light_sensor_mode="realistic"):
		# Fastsim setup
		# XML files typically contain relative names (for map) wrt their own path. Make that work
		xml_dir, xml_file = os.path.split(xml_env)
		if(xml_dir == ""):
			xml_dir = "./"
		oldcwd = os.getcwd()
		os.chdir(xml_dir)
		settings = fs.Settings(xml_file)
		os.chdir(oldcwd)
		
		self.map = settings.map()
		self.background = np.expand_dims(netpbmfile.imread(os.path.join(xml_dir, 'maze_hard.pbm')), axis=-1)
		self.background_size = [200, 200]
		self.map_size = [600, 600]
		self.robot = settings.robot()

		if(display):
			self.display = fs.Display(self.map, self.robot)
		else:
			self.display = None
		
		self.maxVel = 4 # Same as in the C++ sferes2 experiment
		
		# Lasers
		lasers = self.robot.get_lasers()
		n_lasers = len(lasers)
		if(n_lasers > 0):
			self.maxSensorRange = lasers[0].get_range() # Assume at least 1 laser ranger
		else:
			self.maxSensorRange = 0.
		
		#Light sensors
		self.ls_mode = light_sensor_mode
		lightsensors = self.robot.get_light_sensors()
		n_lightsensors = len(lightsensors)
		if(n_lightsensors > 0):
			self.maxLightSensorRange = light_sensor_range # Assume at least 1 laser ranger
		else:
			self.maxLightSensorRange = 0.
		
		# State
		self.initPos = self.get_robot_pos()
		self.current_pos = self.get_robot_pos()
		self.old_pos = self.get_robot_pos()

		self.v1_motor_order = 0.
		self.v2_motor_order = 0.
		
		self.goal = self.map.get_goals()
		self.goalPos = [[goal.get_x(), goal.get_y()] for goal in self.goal]
		self.goalRadius = [goal.get_diam()/2. for goal in self.goal]

		self.observation_space = spaces.Box(low=np.array([0.]*n_lasers + [0.]*2 + [0. if (self.ls_mode == "realistic") else -1]*n_lightsensors), high=np.array([self.maxSensorRange]*n_lasers + [1. if (self.ls_mode == "realistic") else self.maxLightSensorRange]*2 + [1.]*n_lightsensors, dtype=np.float32))
		self.action_space = spaces.Box(low=-self.maxVel, high=self.maxVel, shape=(2,), dtype=np.float32)
		self.max_steps = 2000
		self.step_count = 0
		# Reward
		self.reward_area = None
		if(reward_func not in reward_functions):
			raise RuntimeError("Unknown reward '%s'" % str(reward_func))
		else:
			self.reward_func = reward_functions[reward_func]
		
	def enable_display(self):
		if not self.display:
			self.display = fs.Display(self.map, self.robot)
			self.display.update()

	def disable_display(self):
		if self.display:
			del self.display
			self.display = None

	def get_robot_pos(self):
		pos = self.robot.get_pos()
		return [pos.x(), pos.y(), pos.theta()]

	def get_laserranges(self):
		out = list()
		for l in self.robot.get_lasers():
			r = l.get_dist()
			if r < 0:
				out.append(self.maxSensorRange)
			else:
				out.append(np.clip(r,0.,self.maxSensorRange))
		return out

	def get_lightsensors(self):
		out = list()
		for ls in self.robot.get_light_sensors():
			r = ls.get_distance()
			if(self.ls_mode == "realistic"):
				# r is -1 if no light, dist if light detected 
				# We want the output to be a "realistic" light sensor :
				# - 0 if no light (no target in field or out of range)
				# - (1-d/maxRange)**2 if light detectted
				if((r < 0) or (r > self.maxLightSensorRange)): # Out of range, or not in angular range
					out.append(0.)
				else:
					out.append((1. - r/self.maxLightSensorRange)**2)
			elif(self.ls_mode == "raw"):
				out.append(r)
			else:
				raise RuntimeError("Unknown LS mode: %s" % self.ls_mode)

		return out

	def get_bumpers(self):
		return [float(self.robot.get_left_bumper()), float(self.robot.get_right_bumper())]

	def get_all_sensors(self):
		return self.get_laserranges() + self.get_bumpers() + self.get_lightsensors()

	def step(self, action):
		self.step_count += 1

		# Action is: [leftWheelVel, rightWheelVel]
		[v1, v2] = action
		
		self.v1_motor_order = np.clip(v1,-self.maxVel,self.maxVel)
		self.v2_motor_order = np.clip(v2,-self.maxVel,self.maxVel)
		
		self.robot.move(self.v1_motor_order, self.v2_motor_order, self.map, sticky_walls)

		sensors = self.get_all_sensors()
		reward = self._get_reward()
		
		self.old_pos = self.current_pos
		self.current_pos = self.get_robot_pos()

		if self.step_count >= self.max_steps:
			episode_over = True
		else:
			episode_over = False

		dist_obj = [dist(self.current_pos, goal) for goal in self.goalPos]
		return sensors, reward, episode_over, {"dist_obj":dist_obj, "robot_pos":self.current_pos, "rew_area": self.reward_area}

	def _get_reward(self):
		return self.reward_func(self) # Use reward extraction function
		
	def reset(self):
		p = fs.Posture(*self.initPos)
		self.robot.set_pos(p)
		self.reward_area = None
		self.step_count = 0
		self.current_pos = self.get_robot_pos()
		self.v1_motor_order = 0.
		self.v2_motor_order = 0.
		return self.get_all_sensors()

	def render(self, mode='human', close=False):
		if self.display:
			self.display.update()
		if self.display is None or mode=='rgb_array':
			array = np.repeat(self.background, 3, axis=-1)
			array = np.where((array == 0) | (array == 1), array^1, array) # Flips 0 and 1
			scale = np.array(self.background_size)/np.array(self.map_size)
			for goal_idx, goal in enumerate(self.goalPos):
				pixels_coord = np.array(goal) * scale
				radius = self.goalRadius[goal_idx] * scale[0]
				rr, cc = draw.circle(pixels_coord[1], pixels_coord[0], radius=radius, shape=array.shape)
				array[rr.astype(int), cc.astype(int)] = np.array([1, 0, 0])

			pixels_coord = np.array(self.current_pos[:2]) * scale
			rr, cc = draw.circle(pixels_coord[1], pixels_coord[0], radius=2, shape=array.shape)
			array[rr, cc] = np.array([0, 0, 1])  # Paint circle in blue
			return array * 255
		pass

	def close(self):
		self.disable_display()
		del self.robot
		del self.map
		pass