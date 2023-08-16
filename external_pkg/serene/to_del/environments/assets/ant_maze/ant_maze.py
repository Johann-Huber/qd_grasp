import numpy as np
import os

from gym import utils
from gym.spaces import Dict, Box
from gym.utils import EzPickle
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding

class AntObstaclesBigEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.ts = 0
        self.goal = [np.array([35, -25]),
                     np.array([-25, 35])]
        self.goalRadius = [3, 3]
        self.max_ts = 3000
        self.reward_area = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dir_path, 'xmls/ant_obstaclesbig2.xml'), 5)
        utils.EzPickle.__init__(self)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        return [seed]

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        self.ts += 1
        done = False
        reward = 0.

        if self.ts >= self.max_ts:
            done = True
            for goal_idx, goal in enumerate(self.goal):
                dist = np.linalg.norm(self.data.qpos[:2] - goal)
                if dist <= self.goalRadius[goal_idx]:
                    reward = (self.goalRadius[goal_idx] - dist)/self.goalRadius[goal_idx]
                    self.reward_area = goal_idx

        ob = self._get_obs()

        return ob, reward, done, dict(bc=self.data.qpos[:2],
                                      x_pos=self.data.qpos[0],
                                      x_position=self.data.qpos[0],
                                      y_position=self.data.qpos[1],
                                      rew_area=self.reward_area)

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qpos[:2] = (qpos[:2] - 5) / 70
        return np.concatenate([
            qpos,
            self.data.qvel.flat,
        ])

    def reset_model(self):
        self.reward_area = None
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.ts = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[0] = 4.2
        self.viewer.cam.lookat[1] = 0
        # self.viewer.opengl_context.set_buffer_size(4024, 4024)