
import pdb

import numpy as np
from pathlib import Path
import itertools
from itertools import combinations
import utils.constants as consts

from gym_envs.envs.src.utils import get_simulation_table_height
from gym_envs.envs.src.robot_grasping import RobotGrasping
from gym_envs.envs.src.xacro import _process
import gym_envs
import gym_envs.envs.src.env_constants as env_consts
import gym_envs.envs.src.robots.ur5_based.ur5_consts as u5_consts
import gym_envs.envs.src.robots.ur5_based.ur5_sih_schunk.ur5_sih_consts as u5sih_consts



"""
    # ---------------------------------------------------------------------------------------- #
    #                              UR-5 ARM + SCHUNK SIH HAND
    # ---------------------------------------------------------------------------------------- #
"""


def generate_urdf_from_xacro(root_3d_models_robots, urdf):
    # create the file if doesn't exist (xacro to urdf conversion)
    xacro2urdf_kwargs = dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1, mappings={})
    _process(
        root_3d_models_robots / u5sih_consts.UR5_SIH_SCHUNK_RELATIVE_PATH_XACRO,
        xacro2urdf_kwargs
    )


def init_urdf_ur5_sih_schunk():
    root_3d_models_robots = \
        Path(gym_envs.__file__).resolve().parent / env_consts.GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS

    urdf = Path(root_3d_models_robots / u5sih_consts.UR5_SIH_SCHUNK_RELATIVE_PATH_GENERATED_URDF)

    urdf.parent.mkdir(exist_ok=True)
    if not urdf.is_file():
        generate_urdf_from_xacro(root_3d_models_robots=root_3d_models_robots, urdf=urdf)

    return urdf


class UR5SihSchunk(RobotGrasping):
    def __init__(self, **kwargs):

        ur5_sih_schunk_kwargs = {
            'robot_class': self._load_model,
            'object_position': u5sih_consts.DEFAULT_OBJECT_POSE_XYZ,
            'object_xyzw': u5sih_consts.DEFAULT_OBJECT_ORIENT_XYZW,
            'table_height': env_consts.TABLE_HEIGHT,
            'joint_ids': self._get_joint_ids(),
            'contact_ids': self._get_all_gripper_ids(),
            'disabled_obj_robot_contact_ids': self._get_all_arm_ids(),
            'n_control_gripper': self._get_n_dof_gripper(),
            'end_effector_id': u5sih_consts.SCHUNK_SIH_RIGHT_END_EFFECTOR_JOINT_ID,
            'center_workspace': u5sih_consts.UR5_SIH_SCHUNK_CENTER_WORKSPACE,
            'ws_radius': u5sih_consts.UR5_SIH_SCHUNK_WORKSPACE_RADIUS,
            'disabled_collision_pair': self._get_disabled_collision_pair(),
            'change_dynamics': u5sih_consts.CHANGE_DYNAMICS_DICT,
            'allowed_collision_pair': self._get_allowed_collision_pairs(),
            'is_there_primitive_gene': True,
            'table_label': env_consts.TableLabel.UR5_TABLE
        }

        self.i_action_grip_close = -2
        self.i_action_primitive_label = -1
        self.n_dof_arm = self._get_n_dof_arm()

        super().__init__(
            **ur5_sih_schunk_kwargs,
            **kwargs,
        )

    def get_arm_ids(self):
        return [j_id for j_id in u5sih_consts.UR5_ARM_JOINT_ID_STATUS]

    def get_hand_ids(self):
        return [j_id for j_id in u5sih_consts.SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS]
    '''
    def get_table_id(self):
        assert u5sih_consts.BONN_SCENE_RELATED_JOINTS[22]['name'] == 'table_surface_joint'
        return [22]
    '''

    def _get_allowed_collision_pairs(self):
        autocollision_wall_ids = list(combinations(self._get_wall_ids(), 2))
        robot_table_collisions_ids = []
        hand_walls_collisions_ids = list(itertools.product(self.get_hand_ids(), self._get_wall_ids()))
        return autocollision_wall_ids + robot_table_collisions_ids + hand_walls_collisions_ids

    def _get_disabled_collision_pair(self):
        within_hand_collision_pairs = list(combinations(self._get_all_gripper_ids(), 2))
        wrist_ids = (3, 4, 5)
        base_link_hand_ids = (9, 10)
        wrist_hand_base_collision_pairs = list(itertools.product(wrist_ids, base_link_hand_ids))
        return within_hand_collision_pairs + wrist_hand_base_collision_pairs

    def step(self, action=None):

        assert action is not None
        assert len(action) == self.n_actions
        assert len(action) == 8  # temp

        # Update info
        self.info['closed gripper'] = self._is_gripper_closed(action)

        # Convert action to a gym-grasp compatible command
        gripper_command = self._get_gripper_command_primitives(
            action_gripper_genome_val=action[self.i_action_grip_close],
            primitive_genome_val=action[self.i_action_primitive_label],
        )
        arm_command = action[:self.i_action_grip_close]
        robot_command = np.hstack([arm_command, gripper_command])

        # Send the command to the robot
        return super().step(robot_command)

    def _is_gripper_closed(self, action):
        is_closed = action[self.i_action_grip_close] < 0
        return is_closed

    def _get_all_gripper_ids(self):
        return [j_id for j_id in u5sih_consts.SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS]

    def _get_all_arm_ids(self):
        return [j_id for j_id in u5sih_consts.UR5_ARM_JOINT_ID_STATUS]

    def _get_wall_ids(self):
        return [j_id for j_id in u5sih_consts.BONN_SCENE_RELATED_JOINTS]

    #def _update_info_is_success(self):
    #    # is_success : is the robot holding the object for some steps
    #    #print(f'rwd_cum={self.reward_cumulated} | is_scs={self.reward_cumulated > (30 / self.steps_to_roll)}')
    #    self.info['is_success'] = self.reward_cumulated > (30 / self.steps_to_roll)

    def get_fingers(self, x):
        # Format when giving only controllable joint to robot_grasping:
        th_inter_to_th_distal_j_id = [19, 20]  # rotation of the last joint of the thumb is opposite sided
        return np.array(
            [x if j_id not in th_inter_to_th_distal_j_id else -x
             for j_id in u5sih_consts.SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS
             if u5sih_consts.SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
             ]
        )

    def _get_gripper_command_primitives(self, action_gripper_genome_val, primitive_genome_val):

        assert action_gripper_genome_val in [-1, 1]

        is_closing = action_gripper_genome_val == -1

        if not is_closing:
            return self.get_fingers(action_gripper_genome_val)

        else:
            discarded_j_ids = u5sih_consts.DISCARDED_J_IDS_CLOSING_PRIMITIVES[primitive_genome_val]

            # Format when giving only controllable joint to robot_grasping:
            th_inter_to_th_distal_j_id = [19, 20]  # rotation of the last joint of the thumb is opposite sided # todo depreciated ?
            return np.array(
                [(1 if j_id not in discarded_j_ids else (-1)) * action_gripper_genome_val if j_id not in th_inter_to_th_distal_j_id
                 else (1 if j_id not in discarded_j_ids else (-1)) * -action_gripper_genome_val
                 for j_id in u5sih_consts.SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS
                 if u5sih_consts.SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
                 ]
            )

    def _load_model(self):

        urdf = init_urdf_ur5_sih_schunk()

        table_height = env_consts.TABLE_HEIGHT
        floor_table_pos_z = get_simulation_table_height(table_height)
        robot_basis_offset_z = -0.01
        robot_pos_z_on_table = consts.REAL_SCENE_TABLE_TOP + floor_table_pos_z + robot_basis_offset_z

        robot_body_id = self.bullet_client.loadURDF(
            str(urdf),
            basePosition=[0.6, 0.2, robot_pos_z_on_table],  # default : [0, -0.5, -0.5]
            baseOrientation=[0., 0., 0., 1.],
            useFixedBase=True,
            flags=self.bullet_client.URDF_USE_SELF_COLLISION
        )

        return robot_body_id

    def _get_arm_controllable_joint_ids(self):
        return u5sih_consts.UR5_ARM_CONTROLLABLE_JOINT_IDS

    def _get_gripper_controllable_joint_ids(self):
        return u5sih_consts.SIH_SCHUNK_HAND_CONTROLLABLE_JOINT_IDS

    def _get_arm_controlled_joint_ids(self):
        return u5sih_consts.UR5_ARM_CONTROLLED_JOINT_IDS

    def _get_gripper_controlled_joint_ids(self):
        return u5sih_consts.SIH_SCHUNK_HAND_CONTROLLED_JOINT_IDS

    def _get_n_dof_arm(self):
        return len(self._get_arm_controlled_joint_ids())

    def _get_n_dof_gripper(self):
        return len(self._get_gripper_controlled_joint_ids())

    def _get_controllable_joint_ids(self):
        # controllable: what joints the urdf allows us to control
        return self._get_arm_controllable_joint_ids() + self._get_gripper_controllable_joint_ids()

    def _get_joint_ids(self):
        # controlled: what do we work with, among the controllable joints
        return self._get_arm_controlled_joint_ids() + self._get_gripper_controlled_joint_ids()

    def _get_rest_poses(self):
        return u5sih_consts.REST_POSE_ARM_JOINTS

    def _reset_hand(self):
        for j_id, pos in u5sih_consts.HAND_DEFAULT_JOINT_STATES.items():
            self.bullet_client.resetJointState(self.robot_id, j_id, targetValue=pos)

    def _set_robot_default_state(self):
        for j_id, pos in u5sih_consts.DEFAULT_JOINT_STATES.items():
            self.bullet_client.resetJointState(self.robot_id, j_id, targetValue=pos)

    def _set_robot_manually_fixed_joint_states(self):
        for j_id, pos in u5sih_consts.MANUALLY_SET_JOINT_STATES.items():
            self.bullet_client.resetJointState(self.robot_id, j_id, targetValue=pos)

    def _reset_robot(self):
        self._set_robot_default_state()
        self._set_robot_manually_fixed_joint_states()




