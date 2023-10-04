import numpy as np
from pathlib import Path

import utils.constants as consts
import gym_envs.envs.src.env_constants as env_consts

from gym_envs.envs.src.utils import get_simulation_table_height

# todo : make each attr explicitly named in the constructor (even init at None)


class BulletSimTable:
    def __init__(self, bullet_client, table_height, table_label):

        self.table_height = None  # table height (in m)
        self.table_pos = None  # table position
        self.table_x_size, self.table_y_size = None, None  # table dimensions
        self.table_id = None  # body's id associated to the table in the simulation

        self._init_attributes(bullet_client=bullet_client, table_height=table_height, table_label=table_label)

    def _init_attributes(self, bullet_client, table_height, table_label):

        self.table_height = table_height

        if self.table_height is not None:

            self.table_pos = self._get_table_pos(table_label)
            table_urdf_path = self._get_table_urdf_path(table_label)

            self.table_id = bullet_client.loadURDF(
                table_urdf_path,
                basePosition=self.table_pos,
                baseOrientation=consts.BULLET_TABLE_BASE_ORIENTATION,
                useFixedBase=True
            )
        else:
            self.table_pos, self.table_id = None, None
        self.table_x_size, self.table_y_size = 1.5, 1

    def _get_table_pos(self, table_label):
        table_pos_z = get_simulation_table_height(self.table_height)
        if table_label == env_consts.TableLabel.STANDARD_TABLE:
            return np.array([0, 0.4 - 0.2, table_pos_z])
        elif table_label == env_consts.TableLabel.UR5_TABLE:
            return np.array([0.6, 0.4 - 0.2, table_pos_z])
        else:
            raise RuntimeError(f'Missing table link for {table_label}')

    def _get_table_urdf_path(self, table_label):
        if table_label == env_consts.TableLabel.STANDARD_TABLE:
            return consts.BULLET_TABLE_URDF_FILE_RPATH_REAL_SCENE
        elif table_label == env_consts.TableLabel.UR5_TABLE:
            return consts.BULLET_UR5_TABLE_URDF_FILE_RPATH_REAL_SCENE
        else:
            raise RuntimeError(f'Missing table link for {table_label}')







