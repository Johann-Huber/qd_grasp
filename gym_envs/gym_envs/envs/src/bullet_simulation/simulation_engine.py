import pdb

import numpy as np
import os
from pathlib import Path

import utils.constants as consts
import gym_envs.envs.src.env_constants as env_consts

from gym_envs.envs.src.bullet_simulation.entities.bullet_sim_object import BulletSimObject
from gym_envs.envs.src.bullet_simulation.entities.bullet_sim_table import BulletSimTable
from gym_envs.envs.src.bullet_simulation.entities.bullet_sim_robot import BulletSimRobot
from gym_envs.envs.src.bullet_simulation.entities.bullet_sim_plane import BulletSimPlane


class SimulationEngine:
    def __init__(
            self,
            bullet_client,
            n_dof_arm,
            table_height,
            robot_class,
            joint_ids,
            end_effector_id,
            n_control_gripper,
            ws_radius,
            center_workspace,
            change_dynamics={},
            steps_to_roll=consts.BULLET_DEFAULT_N_STEPS_TO_ROLL,
            object_name=None,
            object_position=consts.BULLET_OBJECT_DEFAULT_POSITION,
            object_xyzw=consts.BULLET_OBJECT_DEFAULT_ORIENTATION,
            initial_state=consts.ENV_DEFAULT_INIT_STATE,
            contact_ids=[],
            allowed_collision_pair=[],
            disabled_collision_pair=[],
            disabled_obj_robot_contact_ids=[],
            is_there_primitive_gene=None,
            n_body_gripper=0,
            controller_type=None,
            table_label=env_consts.TableLabel.STANDARD_TABLE,
    ):

        self.steps_to_roll = None  # number of p.stepSimulation iterations between two action steps
        self.init_state_p_file_root = None  # local save of bullet sim config for quick reinitialization
        self.bullet_sim_plane = None  # manage bullet simulation ground
        self.bullet_sim_table = None  # manage bullet simulation table
        self.bullet_sim_obj = None  # manage bullet simulation object to grasp
        self.bullet_sim_robot = None  # manage bullet simulation robot
        self._do_noise_joint_states = None  #  whether to add noise to joint pose (open loop actions : 7dof pos) or not
        self.reward_cumulated = None  # cumulated rwd (used to identify if the grasp is stable: hold for some steps)

        self._init_attributes(
            bullet_client=bullet_client,
            n_dof_arm=n_dof_arm,
            table_height=table_height,
            robot_class=robot_class,
            joint_ids=joint_ids,
            end_effector_id=end_effector_id,
            n_control_gripper=n_control_gripper,
            ws_radius=ws_radius,
            center_workspace=center_workspace,
            change_dynamics=change_dynamics,
            steps_to_roll=steps_to_roll,
            object_name=object_name,
            object_position=object_position,
            object_xyzw=object_xyzw,
            initial_state=initial_state,
            contact_ids=contact_ids,
            allowed_collision_pair=allowed_collision_pair,
            disabled_collision_pair=disabled_collision_pair,
            disabled_obj_robot_contact_ids=disabled_obj_robot_contact_ids,
            is_there_primitive_gene=is_there_primitive_gene,
            n_body_gripper=n_body_gripper,
            controller_type=controller_type,
            table_label=table_label
        )

    @property
    def plane_id(self):
        return self.bullet_sim_plane.plane_id

    @property
    def obj_id(self):
        return self.bullet_sim_obj.obj_id

    @property
    def table_id(self):
        return self.bullet_sim_table.table_id

    @property
    def robot_id(self):
        return self.bullet_sim_robot.robot_id

    @property
    def n_actions(self):
        return self.bullet_sim_robot.n_actions

    @property
    def end_effector_id(self):
        return self.bullet_sim_robot.end_effector_id

    def _init_attributes(
            self,
            bullet_client,
            n_dof_arm,
            table_height,
            robot_class,
            joint_ids,
            end_effector_id,
            n_control_gripper,
            ws_radius,
            center_workspace,
            change_dynamics,
            steps_to_roll,
            object_name,
            object_position,
            object_xyzw,
            initial_state,
            contact_ids,
            allowed_collision_pair,
            disabled_collision_pair,
            disabled_obj_robot_contact_ids,
            is_there_primitive_gene,
            n_body_gripper,
            controller_type,
            table_label
    ):
        bullet_client.resetSimulation()
        bullet_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        self.steps_to_roll = steps_to_roll
        self.reward_cumulated = 0

        self._init_gravity(bullet_client=bullet_client)

        plane_kwargs = {'bullet_client': bullet_client}
        table_kwargs = {'bullet_client': bullet_client, 'table_height': table_height, 'table_label': table_label}
        object_kwargs = {
            'bullet_client': bullet_client, 'xyz_pose': object_position, 'xyzw_orient': object_xyzw, 'name': object_name
        }
        robot_kwargs = {
            'bullet_client': bullet_client, 'robot_class': robot_class, 'joint_ids': joint_ids,
            'end_effector_id': end_effector_id, 'n_control_gripper': n_control_gripper, 'ws_radius': ws_radius,
            'center_workspace': center_workspace, 'contact_ids': contact_ids,
            'allowed_collision_pair': allowed_collision_pair, 'disabled_collision_pair': disabled_collision_pair,
            'initial_state': initial_state, 'disabled_obj_robot_contact_ids': disabled_obj_robot_contact_ids,
            'is_there_primitive_gene': is_there_primitive_gene, 'n_body_gripper': n_body_gripper,
            'controller_type': controller_type, 'change_dynamics': change_dynamics, 'n_dof_arm': n_dof_arm
        }

        self.bullet_sim_plane = BulletSimPlane(**plane_kwargs)
        self.bullet_sim_table = BulletSimTable(**table_kwargs)
        self.bullet_sim_obj = BulletSimObject(**object_kwargs)

        self._run_stabilization_steps(bullet_client=bullet_client)

        self.bullet_sim_robot = BulletSimRobot(**robot_kwargs)

        self._init_default_infos()

    def _run_stabilization_steps(self, bullet_client):
        print('n_stabilization_steps=', consts.N_ITER_STABILIZING_SIM_DEFAULT_ROBOT_GRASP)
        self._update_n_step_sim(bullet_client=bullet_client, n_steps=consts.N_ITER_STABILIZING_SIM_DEFAULT_ROBOT_GRASP)

    def _update_n_step_sim(self, bullet_client, n_steps):
        for _ in range(n_steps):
            bullet_client.stepSimulation()

    def _init_gravity(self, bullet_client):
        # Set gravity
        if consts.GRAVITY_FLG:
            bullet_client.setGravity(0., 0., -9.81)

    def _init_default_infos(self):
        self.info = {
            'closed gripper': False,
            'contact object table': (),
            'contact robot table': (),
            'joint reaction forces': np.zeros(self.bullet_sim_robot.n_joints),  # todo: en a-t-on besoin ?
            'applied joint motor torques': np.zeros(self.bullet_sim_robot.n_joints),  # todo: en a-t-on besoin ?
            'joint positions': np.zeros(self.bullet_sim_robot.n_joints),  # todo: en a-t-on besoin ?
            'joint velocities': np.zeros(self.bullet_sim_robot.n_joints),  # todo: en a-t-on besoin ?
            'end effector position': np.zeros(3),
            'end effector xyzw': np.zeros(4),
            'end effector linear velocity': np.zeros(3),
            'end effector angular velocity': np.zeros(3),
        }

    def init_local_sim_save(self, bullet_client):

        save_folder_root = os.getcwd() + '/tmp'

        self.init_state_p_file_root = save_folder_root + "/init_state.bullet"
        local_pid_str = str(os.getpid())

        Path(save_folder_root).mkdir(exist_ok=True)

        # Make sure each worker (cpu core) has its own local save
        init_state_p_local_pid_file = self.init_state_p_file_root + '_' + local_pid_str
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(f'dir_path = {dir_path}')
        print('trying to save = ', init_state_p_local_pid_file)
        bullet_client.saveBullet(init_state_p_local_pid_file)
        print(f'init_state_p_local_pid_file={init_state_p_local_pid_file} successfully saved.')

        # Create a unique reference file that is erased each time. Usefull to rely on a unique file if necessary (i.e.
        # in plot_trajectory, that parallelize re-evaluation with Pool and display on a single cpu the results)
        init_state_p_local_pid_file = self.init_state_p_file_root
        bullet_client.saveBullet(init_state_p_local_pid_file)

    def _clip_action(self, action):
        return np.clip(action, -consts.ACTION_UPPER_BOUND, consts.ACTION_UPPER_BOUND)

    def _perturbate_joint_states(self, action):
        noise = np.random.normal(
            size=self.bullet_sim_robot.n_dof_arm,
            loc=consts.NOISE_JOINT_STATES_MU,
            scale=consts.NOISE_JOINT_STATES_SIGMA
        )
        action[:self.bullet_sim_robot.n_dof_arm] += noise
        return action

    def apply_action_to_sim(self, bullet_client, action):

        if self._do_noise_joint_states:
            action = self._perturbate_joint_states(action=action)

        action = self._clip_action(action)
        self._apply_action_to_sim_joints(bullet_client=bullet_client, action=action)
        self._update_n_step_sim(bullet_client=bullet_client, n_steps=self.steps_to_roll)

        observation = self._get_obs()
        info = self._get_info(bullet_client=bullet_client)
        reward = self._get_reward(info)
        done = self._is_done()

        self._update_reward(reward)
        info = self._update_info_is_success(info)

        return observation, reward, done, info

    def _get_obs(self):
        return consts.BULLET_DUMMY_OBS  # open-loop trajectories: observations are not used in the code

    def _is_done(self):
        return False  # not used but kept for consistency (fixed episode length + additional steps for stability check)

    def _get_info(self, bullet_client):
        #  todo : peut encore être amélioré. Initialization plus propre? Que peut-on factoriser ?
        info = {
            'closed gripper': False,
            'contact object table': (),
            'contact robot table': (),
            'joint reaction forces': np.zeros(self.bullet_sim_robot.n_joints),
            'applied joint motor torques': np.zeros(self.bullet_sim_robot.n_joints),
            'joint positions': np.zeros(self.bullet_sim_robot.n_joints),
            'joint velocities': np.zeros(self.bullet_sim_robot.n_joints),
            'end effector position': np.zeros(3),
            'end effector xyzw': np.zeros(4),
            'end effector linear velocity': np.zeros(3),
            'end effector angular velocity': np.zeros(3),
        }
        info = self.bullet_sim_obj.update_infos(bullet_client=bullet_client, info=info)
        info = self.bullet_sim_robot.update_robot_infos(bullet_client=bullet_client, info=info)
        info = self.bullet_sim_robot.update_contact_infos(
            bullet_client=bullet_client, info=info, obj_id=self.obj_id, plane_id=self.plane_id, table_id=self.table_id
        )
        return info

    def _update_reward(self, reward):
        self.reward_cumulated += reward

    def _update_info_is_success(self, info):
        # is_success : is the robot holding the object for some steps
        # todo : 100 pour tous les robots, 30 pour ur5_shunk
        info['is_success'] = self.reward_cumulated > (100 / self.steps_to_roll)
        return info

    def _get_reward(self, info):
        no_obj_table_contact = len(info['contact object table']) == 0
        no_obj_ground_contact = len(info['contact object plane']) == 0 if self.obj_id else False
        no_rob_table_contact = len(info['contact robot table']) == 0
        is_there_rob_obj_contact = info['touch']

        return no_obj_table_contact and no_obj_ground_contact and no_rob_table_contact and is_there_rob_obj_contact

    def _apply_action_to_sim_joints(self, bullet_client, action):
        try:
            assert len(action) == len(self.bullet_sim_robot.joint_ids)
            assert len(action) == len(self.bullet_sim_robot.max_velocity)
            assert len(action) == len(self.bullet_sim_robot.max_force)
            assert len(action) == len(self.bullet_sim_robot.upper_limits)
            assert len(action) == len(self.bullet_sim_robot.lower_limits)
        except:
            pdb.set_trace()

        iter_control = zip(
            self.bullet_sim_robot.joint_ids, action,
            self.bullet_sim_robot.max_velocity,
            self.bullet_sim_robot.max_force,
            self.bullet_sim_robot.upper_limits,
            self.bullet_sim_robot.lower_limits
        )
        for id, joint_a, max_vel, max_f, up_lim, low_lim in iter_control:
            target_pos = low_lim + (joint_a + 1) / 2 * (up_lim - low_lim)
            bullet_client.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=id,
                controlMode=bullet_client.POSITION_CONTROL,
                targetPosition=target_pos,
                maxVelocity=max_vel,
                force=max_f
            )

    def get_state(self, bullet_client):
        """Returns the unnormalized object position and joint positions.
        The returned dict can be passed to initialize another environment."""

        pos, qua = bullet_client.getBasePositionAndOrientation(self.obj_id) if self.obj_id is not None else (None, None)
        joint_positions = [s[0] for s in bullet_client.getJointStates(self.robot_id, self.bullet_sim_robot.joint_ids)]
        env_state = {'object_position': pos, 'object_xyzw': qua, 'joint_positions': joint_positions}
        return env_state

    def get_joint_state(self, bullet_client, position=True, normalized=True):
        """ Return (un)normalized joint positions (velocities) without the gripper"""
        return self.bullet_sim_robot.get_joint_state(
            bullet_client=bullet_client,
            position=position,
            normalized=normalized,
        )

    def get_joints_poses_from_ik(self, bullet_client, pos, or_pry, normalized, arm_controllable_joint_ids, rest_poses,
                                 controllable_joint_ids):
        return self.bullet_sim_robot.get_joints_poses_from_ik(
            bullet_client=bullet_client,
            pos=pos,
            or_pry=or_pry,
            normalized=normalized,
            arm_controllable_joint_ids=arm_controllable_joint_ids,
            rest_poses=rest_poses,
            controllable_joint_ids=controllable_joint_ids
        )

    def reset(self, do_noise_joints_states):
        self._do_noise_joint_states = do_noise_joints_states
        self.reward_cumulated = 0
        return self._get_obs()

    def reset_object_pose(self, bullet_client, obj_init_state_offset):
        self.bullet_sim_obj.reset_object_pose(
            bullet_client=bullet_client, obj_init_state_offset=obj_init_state_offset
        )

    def perturbate_dynamics(self, bullet_client):

        noisy_rolling_friction = np.random.uniform(
            low=consts.ROLLING_FRICTION_DOMAIN_RAND_MIN_VALUE,
            high=consts.ROLLING_FRICTION_DOMAIN_RAND_MAX_VALUE
        )
        noisy_spinning_friction = np.random.uniform(
            low=consts.SPINNING_FRICTION_DOMAIN_RAND_MIN_VALUE,
            high=consts.SPINNING_FRICTION_DOMAIN_RAND_MAX_VALUE
        )
        bullet_client.changeDynamics(
            bodyUniqueId=self.obj_id, linkIndex=-1,
            rollingFriction=noisy_rolling_friction,
            spinningFriction=noisy_spinning_friction
        )

