import pdb
import os
from gym import Env, spaces
from time import sleep
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from pathlib import Path
import weakref
import functools
import numpy as np
from collections import OrderedDict
import time
import torch as th
import pybullet as p
import glob
import gym_envs
import utils.constants as consts
from gym_envs.envs.utils import get_simulation_table_height
import gym_envs.envs.env_constants as env_consts

from typing import Dict, List, Tuple, Sequence, Callable, Any, Union, Optional
try:
    import numpy.typing as npt
    ArrayLike = npt.ArrayLike
except:
    ArrayLike = Any


class RobotGrasping(Env):

    metadata = {'render.modes': ['human', 'rgb_array', 'rgba_array']}

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    def __init__(self,
        robot_class,
        table_height,
        end_effector_id,
        joint_ids,
        n_control_gripper,
        center_workspace,
        ws_radius,
        object_name=None,
        display=consts.BULLET_DEFAULT_DISPLAY_FLG,
        gripper_display=consts.BULLET_DEFAULT_GRIP_DISPLAY_FLG,
        steps_to_roll=consts.BULLET_DEFAULT_N_STEPS_TO_ROLL,
        object_position=consts.BULLET_OBJECT_DEFAULT_POSITION,
        object_xyzw=consts.BULLET_OBJECT_DEFAULT_ORIENTATION,
        initial_state=consts.ENV_DEFAULT_INIT_STATE,
        contact_ids=[],
        disabled_collision_pair=[],
        allowed_collision_pair=[],
        disabled_obj_robot_contact_ids=[],
        change_dynamics={},
        fixed_arm=None,
        is_there_primitive_gene=None,
        run_name=None,
        n_body_gripper=0,
        controller_type=None,
    ):

        # ----------------------------------------- #
        #               Definitions
        # ----------------------------------------- #

        # Bullet client
        self.p = None  # bullet physics client
        self.physicsClientId = None  # bullet physics client id

        # Export path
        self._run_name = run_name

        # Rendering
        self.display = None  # whether to display steps
        self.gripper_display = None  # whether to display the end effector trajectory
        self.camera = None

        # Simulation
        self.change_dynamics = None  # the key is the robot link id, the value is the args passed to p.changeDynamics
        self.time_step = None
        self.steps_to_roll = None  # number of p.stepSimulation iterations between two action steps
        self.penetration = None  # is there penetration between robot and object (simulation issue : discarded traj)

        # --- Plane
        self.plane_id = None  # bullet id associated with the simulated ground

        # --- Table
        self.table_height = None  # table height (in m)
        self.table_pos = None  # table position
        self.table_id = None  # body's id associated to the table in the simulation
        self.table_x_size, self.table_y_size = None, None  # table dimensions

        # --- Object
        self.object_name = None  # string describing the targeted object
        self.object_position = None  # object position (in m)
        self.object_xyzw = None  # object orientation (in quaterions)
        self.frictions = None
        self.obj_id = None  # bullet body id corresponding to current object 
        self._initial_stabilized_object_pos = None  # (pos, qua) of the stabilized object on the table
        #self.obj_length = None # 

        # --- Robot
        self.robot_class = None  # function that can be called to create the bullet instance corresponding to the robot
        self.robot_id = None  # bullet id associated with the simulated robot
        self.controller_type = None  # str describing the type of the controller
        self.joint_ids = None  # ids of controllable joints, ending with the gripper
        self.end_effector_id = None  # link id of the end effector
        self.n_body_gripper = None  # number of controlled joint belonging to the body (=0 for body-less manipulators)
        self.n_control_gripper = None  # number of controlled joint belonging to the gripper
        self.n_controllable_joints = None # number of controllable (w.r.t. urdf) joints
        self.ws_radius = None  # radius of the workspace
        self.center_workspace = None  # position of the center of the sphere
        self.contact_ids = None  # robots' link ids with allowed contacts
        self.allowed_collision_pair = None  # 2D array (-1,2), list of pair of link id (int) allowed in autocollision
        self.initial_state = None  # robot initial state (used in robot_env)
        self._do_noise_joints_pose = None  # whether to add noise to joint pose (open loop actions : 7dof pos) or not
        self.disabled_obj_robot_contact_ids = None  # list of robot joint id considered as autocollided if the obj touches it

        self.n_joints = None  # number of joint positions
        self.n_actions = None  # len of the action vector
        self.center_workspace_cartesian = None
        self.center_workspace_robot_frame = None
        self.is_there_primitive_gene = None  # Flag triggered if the genome contains a primitive gene (for dexterous hands, False otherwise)

        self.lower_limits = None  # joint position lower bound
        self.upper_limits = None  # joint position upper bound
        self.max_force = None  # applied torque (on joints) upper bound
        self.max_velocity = None  # joint velocity upper bound
        self.max_acceleration = None # joint acceleration upper bound

        self.observation_space = None  # observation space defined as gym spaces
        self.robot_space = None  # robot space defined as gym spaces
        self.action_space = None  # action space defined as gym spaces


        # Local save
        self.init_state_p_file_root = None  # local save of bullet sim config for quick reinitialization
        self.i_step_last_rolling_step = None  # rolling local save for rolling back when touching the obj

        # Random
        self.rng = np.random.default_rng()  # random generator

        # Rolling variables
        self.info = None
        self.reward_cumulated = 0  # cumulated rwd (used to identify if the grasp is stable, i.e. hold for some steps)
        self.debug_i_step = 0
        self.debug_i_debug_bodies = []

        # Debug monitoring variables
        self.monitor_servoing_debug = {'full_target_pos': [], 'current_pos': []}

        # ----------------------------------------- #
        #              Initializations
        # ----------------------------------------- #

        self._init_bullet_physics_client(display=display)
        self._init_bullet_sim(change_dynamics=change_dynamics, steps_to_roll=steps_to_roll)

        self._init_rendering(
            steps_to_roll=steps_to_roll, gripper_display=gripper_display, display=display
        )

        self._init_ground()
        self._init_gravity()
        self._init_table(table_height=table_height, robot_class=robot_class)

        self._init_object(object_name=object_name, object_position=object_position, object_xyzw=object_xyzw)
        self._run_stabilization_steps(skip_robot_reset=True)
        self._initial_stabilized_object_pos = self.initial_stabilized_object_pos()

        self._init_robot(
            robot_class=robot_class, joint_ids=joint_ids, end_effector_id=end_effector_id,
            n_control_gripper=n_control_gripper, ws_radius=ws_radius, center_workspace=center_workspace,
            contact_ids=contact_ids, allowed_collision_pair=allowed_collision_pair,
            disabled_collision_pair=disabled_collision_pair, initial_state=initial_state,
            disabled_obj_robot_contact_ids=disabled_obj_robot_contact_ids,
            is_there_primitive_gene=is_there_primitive_gene, n_body_gripper=n_body_gripper,
            controller_type=controller_type
        )
        self._init_default_infos()

        self._init_local_sim_save()

    def _init_joint_ids(self, joint_ids):
        assert joint_ids is not None, 'joint_ids cannot be None'
        return np.array(joint_ids, dtype=int)

    def _init_object_name(self, object_name):
        return object_name.strip() if object_name is not None else None

    def _init_object(self, object_name, object_position, object_xyzw):
        self.object_position = object_position
        self.object_xyzw = object_xyzw

        if object_name is None:
            print('Warning : no given object.')
            return

        self.object_name = self._init_object_name(object_name)
        self._load_object_bullet()
        self.frictions = self._init_frictions()

    def _init_robot(self, robot_class, joint_ids, end_effector_id, n_control_gripper, ws_radius, center_workspace,
                    contact_ids, allowed_collision_pair, disabled_collision_pair, initial_state,
                    disabled_obj_robot_contact_ids, is_there_primitive_gene, n_body_gripper, controller_type):

        # Initializing bullet robot
        self.robot_class = robot_class
        self.controller_type = controller_type

        self.robot_id = self.robot_class()

        self._print_all_robot_joint_infos()

        self._reset_robot()

        # Initializing robot hyperparameters
        self.joint_ids = self._init_joint_ids(joint_ids)
        self.end_effector_id = end_effector_id
        self.n_control_gripper = n_control_gripper
        self.n_body_gripper = n_body_gripper
        self.ws_radius = ws_radius
        self.center_workspace = center_workspace
        self.contact_ids = contact_ids
        self.disabled_obj_robot_contact_ids = disabled_obj_robot_contact_ids
        #pdb.set_trace()
        self.allowed_collision_pair = self._init_allowed_collision_pairs(allowed_collision_pair)
        self._do_noise_joints_pose = False  # by default to False, must be changed in reset() if wanted

        n_joints = len(self.joint_ids)
        self.n_joints = n_joints
        self.is_there_primitive_gene = is_there_primitive_gene

        n_action_close_control = 1
        self.n_actions = n_joints - self.n_body_gripper - self.n_control_gripper + n_action_close_control + \
                         int(self.is_there_primitive_gene)

        self.n_controllable_joints = self.n_joints

        self.center_workspace_cartesian = self._init_center_workspace_cartesian()
        self.center_workspace_robot_frame = self._init_center_workspace_robot_frame()
        self._disable_collision_pair(disabled_collision_pair)

        self._init_scene_limits(n_joints)

        self.initial_state = initial_state

        self._init_MDP_spaces()

    def _print_all_robot_joint_infos(self, robot_id=None):
        target_robot_id = robot_id if robot_id is not None else self.robot_id
        n_robot_joints = self.p.getNumJoints(bodyUniqueId=target_robot_id)
        print('-'*50)
        for i_joint in range(n_robot_joints):
            j_infos = self.p.getJointInfo(bodyUniqueId=target_robot_id, jointIndex=i_joint)
            print(f'i_joint={i_joint} | {j_infos}')
        print('-' * 50)

    def _init_rendering(self, steps_to_roll, gripper_display, display):
        self.metadata['video.frames_per_second'] = 240 / steps_to_roll
        self.display = display
        self.gripper_display = gripper_display
        self.camera = self._init_camera(display)

        if display:  # set the camera
            self._init_display()

    def sanity_check_action(self, action):
        assert len(action) == self.n_actions

    def _init_ground(self):
        # Load plan
        offset_plan_z = -1
        self.plane_id = self.p.loadURDF(consts.BULLET_PLANE_URDF_FILE_RPATH, [0, 0, offset_plan_z], useFixedBase=True)

    def _init_gravity(self):
        # Set gravity
        if consts.GRAVITY_FLG:
            self.p.setGravity(0., 0., -9.81)

    def _init_allowed_collision_pairs(self, allowed_collision_pair):
        return [set(c) for c in allowed_collision_pair]

    def _init_camera(self, display):
        cam = consts.CAMERA_DEFAULT_PARAMETERS

        cam['width'] = cam.get('width', 1024)
        cam['height'] = cam.get('height', 1024)
        cam['target'] = cam.get('target', (0, 0, 0))
        cam['distance'] = cam.get('distance', 1)
        cam['yaw'] = cam.get('yaw', 180)
        cam['pitch'] = cam.get('pitch', 0)
        cam['viewMatrix'] = self.p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam['target'],
            distance=cam['distance'],
            yaw=cam['yaw'],
            pitch=cam['pitch'],
            roll=cam.get('roll', 0),
            upAxisIndex=cam.get('upAxisIndex', 2),
        )
        cam['projectionMatrix'] = self.p.computeProjectionMatrixFOV(
            fov=cam.get('fov', 90),
            aspect=cam['width'] / cam['height'],
            nearVal=cam.get('nearVal', 0.1),
            farVal=cam.get('farVal', 10),
        )
        cam['renderer'] = self.p.ER_BULLET_HARDWARE_OPENGL if display else self.p.ER_TINY_RENDERER

        return cam

    def _init_MDP_spaces(self):
        assert self.observation_space is None and self.robot_space is None and self.action_space is None

        obs_upper_bound = [
            1, 1, 1,  # end effector position (robot frame)
            1, 1, 1, 1, 1, 1,  # end effector orientation (robot frame)
            np.inf, np.inf, np.inf,  # end effector linear velocity
            *[1 for i in self.joint_ids],  # joint positions
            # joint velocities can be bounded or unbouded
            *[np.inf for i in self.joint_ids],
            *[1 for i in self.joint_ids],  # joint torque sensors Mz
            1, 1, 1, 1, 1, 1,  # object orientation, 6 coefficients of the rotation matrix
            np.inf, np.inf, np.inf,  # object linear velocity
            np.inf, np.inf, np.inf,  # object angular velocity
            1, 1, 1,  # object position (robot frame)
        ]
        obs_upper_bound = np.array(obs_upper_bound, dtype=np.float32)
        rob_upper_bound = obs_upper_bound[:-15]  # remove components related to the object

        observation_space = spaces.Box(-obs_upper_bound, obs_upper_bound, dtype='float32')
        robot_space = spaces.Box(-rob_upper_bound, rob_upper_bound, dtype='float32')  # robot state only

        action_upper_bound = consts.ACTION_UPPER_BOUND

        action_space = spaces.Box(-action_upper_bound, action_upper_bound, shape=(self.n_actions,), dtype='float32')
        self.observation_space = observation_space
        self.robot_space = robot_space
        self.action_space = action_space

    def _init_default_infos(self):

        self.info = {
            'closed gripper': False,
            'contact object table': (),
            'contact robot table': (),
            'joint reaction forces': np.zeros(self.n_joints),
            'applied joint motor torques': np.zeros(self.n_joints),
            'joint positions': np.zeros(self.n_joints),
            'joint velocities': np.zeros(self.n_joints),
            'end effector position': np.zeros(3),
            'end effector xyzw': np.zeros(4),
            'end effector linear velocity': np.zeros(3),
            'end effector angular velocity': np.zeros(3),
        }

    def _init_bullet_sim(self, change_dynamics, steps_to_roll):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)


        self.change_dynamics = change_dynamics

        self.time_step = self.p.getPhysicsEngineParameters()["fixedTimeStep"]

        self.steps_to_roll = steps_to_roll

    def _init_bullet_physics_client(self, display):
        self.p = BulletClient(connection_mode=p.GUI if display else p.DIRECT)
        self.physicsClientId = self.p._client
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  # add link to urdf files

    def _load_object_bullet(self):
        self.load_object(self.object_name)

    def _init_frictions(self):
        dynamicsInfo = self.p.getDynamicsInfo(self.obj_id, -1)  # save intial friction coefficient of the object
        frictions = {'lateral': dynamicsInfo[1], 'rolling': dynamicsInfo[6], 'spinning': dynamicsInfo[7]}
        return frictions

    def _init_center_workspace_cartesian(self):
        return np.array(
            self.p.getLinkState(self.robot_id, self.center_workspace)[0]
            if isinstance(self.center_workspace, int)
            else self.center_workspace
        )

    def _init_center_workspace_robot_frame(self):
        # Position of center_workspace in the robot frame
        return self.p.multiplyTransforms(
            *self.p.invertTransform(*self.p.getBasePositionAndOrientation(self.robot_id)),
            self.center_workspace_cartesian,
            [0, 0, 0, 1]
        )

    def _disable_collision_pair(self, disabled_collision_pair):
        for contact_point in disabled_collision_pair:
            self.p.setCollisionFilterPair(
                self.robot_id, self.robot_id, contact_point[0], contact_point[1], enableCollision=0
            )

    def _init_table(self, table_height, robot_class):
        self.table_height = table_height

        if self.table_height is not None:

            table_pos_z = get_simulation_table_height(self.table_height)
            if consts.REAL_SCENE:
                # Real scene Y : default (=0.4) - 20 cm
                self.table_pos = np.array([0, 0.4 - 0.2, table_pos_z])
            else:
                self.table_pos = np.array([0, 0.4, table_pos_z])

            table_urdf_path = consts.BULLET_TABLE_URDF_FILE_RPATH_REAL_SCENE if consts.LOCAL_PATH_SIM2REAL_SCENE_FLG \
                else consts.BULLET_TABLE_URDF_FILE_RPATH

            self.table_id = self.p.loadURDF(
                table_urdf_path,
                basePosition=self.table_pos,
                baseOrientation=consts.BULLET_TABLE_BASE_ORIENTATION,
                useFixedBase=True
            )
        else:
            self.table_pos, self.table_id = None, None
        self.table_x_size, self.table_y_size = 1.5, 1


    def _init_display(self):
        assert self.display

        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        self.p.resetDebugVisualizerCamera(
            cameraDistance=self.camera['distance'],
            cameraYaw=self.camera['yaw'],
            cameraPitch=self.camera['pitch'],
            cameraTargetPosition=self.camera['target']
        )
        self.p.resetDebugVisualizerCamera(cameraDistance=self.camera['distance'], cameraYaw=self.camera['yaw'], cameraPitch=self.camera['pitch'], cameraTargetPosition=self.camera['target'])


        if self.gripper_display:
            self.line_width = consts.GRIPPER_DISPLAY_LINE_WIDTH
            self.lines = [
                self.p.addUserDebugLine(
                    [0, 0, 0],
                    end,
                    color,
                    lineWidth=self.line_width,
                    parentObjectUniqueId=self.robot_id,
                    parentLinkIndex=self.end_effector_id)
                for end, color in zip(np.eye(3) * 0.2, np.eye(3))
            ]

    def _print_dynamics(self, from_bullet=False, verbose_str=''):
        if from_bullet:
            j_id = 0
            j_dynamics = self.p.getDynamicsInfo(bodyUniqueId=self.robot_id, linkIndex=j_id)
            mass, lateral_friction, local_inertia_diag, local_inertia_pos, local_inertia_orn, restitution,\
                rolling_friction, spinning_friction, contact_damping, contact_stiffness, body_type, collision_margin \
                = j_dynamics

            print('-' * 10 + verbose_str + '-' * 10)
            print(f'mass={mass}')
            print(f'lateral_friction={lateral_friction}')
            print(f'local_inertia_diag={local_inertia_diag}')
            print(f'local_inertia_pos={local_inertia_pos}')
            print(f'local_inertia_orn={local_inertia_orn}')
            print(f'restitution={restitution}')
            print(f'rolling_friction={rolling_friction}')
            print(f'spinning_friction={spinning_friction}')
            print(f'contact_damping={contact_damping}')
            print(f'contact_stiffness={contact_stiffness}')
            print(f'body_type={body_type}')
            print(f'collision_margin={collision_margin}')
            print('-' * (20 + len(verbose_str) - 1))
        else:
            print('-' * 10 + verbose_str + '-' * 10 )
            print(f'self.lower_limits={self.lower_limits}')
            print(f'self.upper_limits={self.upper_limits}')
            print(f'self.max_velocity={self.max_velocity}')
            print(f'self.max_force={self.max_force}')
            print('-' * (20 + len(verbose_str) - 1))

    def _init_scene_limits(self, n_joints):
        self.lower_limits = np.zeros(n_joints)
        self.upper_limits = np.zeros(n_joints)
        self.max_force = np.zeros(n_joints)
        self.max_velocity = np.zeros(n_joints)

        for i, id in enumerate(self.joint_ids):
            self.lower_limits[i], self.upper_limits[i], self.max_force[i], self.max_velocity[i] = \
                self.p.getJointInfo(self.robot_id, id)[8:12]
            self.p.enableJointForceTorqueSensor(self.robot_id, id)

        # change dynamics
        self._print_dynamics(verbose_str='before modif', from_bullet=True)
        for id, args in self.change_dynamics.items():
            if id in self.joint_ids:  # update limits if needed
                index = np.nonzero(self.joint_ids == id)[0][0]
                if 'jointLowerLimit' in args and 'jointUpperLimit' in args:
                    self.lower_limits[index] = args['jointLowerLimit']
                    self.upper_limits[index] = args['jointUpperLimit']
                if 'maxJointVelocity' in args:
                    self.max_velocity[index] = args['maxJointVelocity']
                if 'jointLimitForce' in args:
                    self.max_force[index] = args['jointLimitForce']

                self.p.changeDynamics(self.robot_id, linkIndex=id, **args)
        self._print_dynamics(verbose_str='after modif', from_bullet=True)

        self.max_force = np.where(self.max_force <= 0, 100, self.max_force)  # replace bad values
        self.max_velocity = np.where(self.max_velocity <= 0, 1, self.max_velocity)
        self.max_acceleration = np.ones(n_joints) * 10  # set maximum acceleration for inverse dynamics

    def _init_local_sim_save(self):

        save_folder_root = os.getcwd() + '/tmp'

        self.init_state_p_file_root = save_folder_root + "/init_state.bullet"
        local_pid_str = str(os.getpid())

        Path(save_folder_root).mkdir(exist_ok=True)

        # Make sure each worker (cpu core) has its own local save
        init_state_p_local_pid_file = self.init_state_p_file_root + '_' + local_pid_str
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(f'dir_path = {dir_path}')
        print('trying to save = ', init_state_p_local_pid_file)
        self.p.saveBullet(init_state_p_local_pid_file)
        print(f'init_state_p_local_pid_file={init_state_p_local_pid_file} successfully saved.')

        # Create a unique reference file that is erased each time. Usefull to rely on a unique file if necessary (i.e.
        # in plot_trajectory, that parallelize re-evaluation with Pool and display on a single cpu the results)
        init_state_p_local_pid_file = self.init_state_p_file_root
        self.p.saveBullet(init_state_p_local_pid_file)

    def _is_gripper_closed(self, action):
        assert False  # must be overloading in subclasses

    def _clip_action(self, action):
        return np.clip(action, -consts.ACTION_UPPER_BOUND, consts.ACTION_UPPER_BOUND)

    def _apply_action_to_sim(self, action):
        self._apply_action_to_sim_joints(action)
        self._update_n_step_sim(self.steps_to_roll)

    def _apply_action_to_sim_cartesian(self, action):
        # Arm movement through IK
        try:
            assert len(action) == 6 + len(self._get_gripper_controllable_joint_ids())
        except:
            pdb.set_trace()

        pos = action[:3]
        or_pry = action[3:6]
        grip_actions = action[6:]

        debug_target_steps2plot = [0, 33, 99, 165]

        debug_plot_target = self.debug_i_step in debug_target_steps2plot
        if debug_plot_target:
            print(f'WAY POINT : {self.debug_i_step}')
        self.debug_i_step += 1
        plot_target = False
        self.apply_cartesian_ik_pose(pos, or_pry=or_pry, plot_target=plot_target)

        # Grip closure through standard joint control
        grip_j_ids = self._get_gripper_controllable_joint_ids()
        assert len(grip_j_ids) == len(grip_actions)

        n_grip_j_ids = len(grip_j_ids)
        max_vel_grip, max_force_grip, upper_limits_grip, lower_limits_grip = \
            self.max_velocity[:n_grip_j_ids], self.max_force[:n_grip_j_ids], \
            self.upper_limits[:n_grip_j_ids], self.lower_limits[:n_grip_j_ids]

        assert len(grip_j_ids) == len(grip_actions) == len(max_vel_grip) == len(max_force_grip) == \
               len(upper_limits_grip) == len(lower_limits_grip)

        iter_control = zip(grip_j_ids, grip_actions, max_vel_grip, max_force_grip, upper_limits_grip,
                           lower_limits_grip)
        for id, joint_a, max_vel, max_f, up_lim, low_lim in iter_control:
            target_pos = low_lim + (joint_a + 1) / 2 * (up_lim - low_lim)
            self.p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=id,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=target_pos,
                maxVelocity=max_vel,
                force=max_f
            )

    def _apply_action_to_sim_joints(self, action):
        try:
            assert len(action) == len(self.joint_ids)
            assert len(action) == len(self.max_velocity)
            assert len(action) == len(self.max_force)
            assert len(action) == len(self.upper_limits)
            assert len(action) == len(self.lower_limits)
        except:
            pdb.set_trace()

        PRINT_TARGET_POS_CURR_POS_DEBUG = False
        if PRINT_TARGET_POS_CURR_POS_DEBUG:
            iter_control = zip(self.joint_ids, action, self.max_velocity, self.max_force, self.upper_limits,
                               self.lower_limits)
            # pdb.set_trace()
            full_target_pos = []
            for id, joint_a, max_vel, max_f, up_lim, low_lim in iter_control:
                target_pos = low_lim + (joint_a + 1) / 2 * (up_lim - low_lim)
                full_target_pos.append(target_pos)
            current_pos = self.get_state()['joint_positions']

            print('-' * 50)
            # print('action=', action)
            print('full_target_pos=', full_target_pos)
            print('current_pos=', current_pos)

            self.monitor_servoing_debug['full_target_pos'].append(full_target_pos)
            self.monitor_servoing_debug['current_pos'].append(current_pos)

        iter_control = zip(self.joint_ids, action, self.max_velocity, self.max_force, self.upper_limits,
                           self.lower_limits)
        for id, joint_a, max_vel, max_f, up_lim, low_lim in iter_control:
            target_pos = low_lim + (joint_a + 1) / 2 * (up_lim - low_lim)
            self.p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=id,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=target_pos,
                maxVelocity=max_vel,
                force=max_f
            )

    def object_has_moved(self):
        assert self.obj_id is not None
        curr_obj_pos, curr_obj_qua = self.p.getBasePositionAndOrientation(self.obj_id)

        init_pos, init_qua = self._initial_stabilized_object_pos
        diff_pos = (np.array(curr_obj_pos) - np.array(init_pos)).sum()
        diff_qua = (np.array(curr_obj_qua) - np.array(init_qua)).sum()

        diff_thresh = 1e-3
        has_moved = diff_pos > diff_thresh or diff_qua > diff_thresh

        return has_moved

    def _get_gripper_controllable_joint_ids(self):
        raise NotImplementedError('Must be overloaded')

    def _update_info(self, action, observation):
        self.info['closed gripper'] = self._is_gripper_closed(action)

        is_obj_initialized = observation.tolist() != consts.BULLET_DUMMY_OBS

        if is_obj_initialized:
            self.info['contact object robot'] = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.robot_id)
            self.info['contact object plane'] = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.plane_id)

        if self.table_id is not None and is_obj_initialized:
            self.info['contact object table'] = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.table_id)
            self.info['contact robot table'] = self.p.getContactPoints(bodyA=self.robot_id, bodyB=self.table_id)

        self.info['touch'], self.info['autocollision'], self.penetration = False, False, False
        self.info['touch_points_obj'], self.info['touch_points_robot'] = [], []

        if is_obj_initialized:
            for c in self.info['contact object robot']:

                self.penetration = self.penetration or c[8] < -0.005  # if contactDistance is negative, there is a penetration, this is bad
                self.info['touch'] = self.info['touch'] or c[4] in self.contact_ids  # the object must touch the gripper

                if c[4] in self.contact_ids:
                    touch_points_on_obj = c[5]
                    touch_points_on_robot = c[6]
                    self.info['touch_points_obj'].append(touch_points_on_obj)
                    self.info['touch_points_robot'].append(touch_points_on_robot)


                if c[4] in self.disabled_obj_robot_contact_ids:
                    self.info['autocollision'] = True
                    break

        self.info['contact robot robot'] = self.p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        for c in self.info['contact robot robot']:
            is_debug_body_touched = sum([int(body_id in set(c[3:5])) for body_id in self.debug_i_debug_bodies]) > 0
            if is_debug_body_touched:
                continue

            if set(c[3:5]) not in self.allowed_collision_pair:
                print('AUTOCOLLIDE : collision pair = ', c[3:5])
                self.info['autocollision'] = True
                break

        self.info['normalized joint pos'] = self.get_joint_state(position=True)
        self.info['terminal_observation'] = observation  # normalized obs

    def _update_gripper_display(self):
        end = np.array(self.p.getMatrixFromQuaternion(self.info['end effector xyzw'])).reshape(3, 3).T @ \
              (np.eye(3) * 0.2) + self.info['end effector position']
        self.lines = [
            self.p.addUserDebugLine(
                self.info['end effector position'], end, color, lineWidth=self.line_width, replaceItemUniqueId=id
            ) for end, color, id in zip(end, np.eye(3), self.lines)
        ]

    def _update_reward(self, reward):
        self.reward_cumulated += reward

    def _update_info_is_success(self):
        # is_success : is the robot holding the object for some steps
        self.info['is_success'] = self.reward_cumulated > (100 / self.steps_to_roll)

    def _get_reward(self):
        no_obj_table_contact = len(self.info['contact object table']) == 0
        no_obj_ground_contact = len(self.info['contact object plane']) == 0 if self.obj_id else False
        no_rob_table_contact = len(self.info['contact robot table']) == 0
        is_there_rob_obj_contact = self.info['touch']
        no_rob_obj_penetration = not self.penetration


        return no_obj_table_contact and no_obj_ground_contact and no_rob_table_contact and is_there_rob_obj_contact \
            and no_rob_obj_penetration

    def step(self, action):
        assert action is not None

        if self._do_noise_joints_pose:
            noise = np.random.normal(size=self.n_dof_arm, loc=0.0, scale=1e-2)
            action[:self.n_dof_arm] += noise

        action = self._clip_action(action)
        self._apply_action_to_sim(action)

        observation = self.get_obs()
        self._update_info(action=action, observation=observation)

        if self.display and self.gripper_display:
            self._update_gripper_display()

        reward = self._get_reward()
        self._update_reward(reward)
        self._update_info_is_success()

        done = False  # not used but kept for consistency (fixed episode length + additional steps for stability check)
        info = {**self.info}  # copy is required to protect self.info

        return observation, reward, done, info

    def load_object(self, obj = None):
        pos = np.array(self.object_position)

        assert isinstance(obj, str)
        urdf = Path(__file__).parent/"objects"/obj/f"{obj}.urdf"
        if not urdf.exists():
            raise ValueError(str(urdf) + " doesn't exist")

        try:
            obj_to_grab_id = self.p.loadURDF(str(urdf), pos, self.object_xyzw, useMaximalCoordinates=True)
        except self.p.error as e:
            raise self.p.error(f"{e}: " + str(urdf))

        self.p.changeDynamics(obj_to_grab_id, -1, spinningFriction=1e-2, rollingFriction=1e-3, lateralFriction=0.5)

        self.obj_id = obj_to_grab_id

    def set_new_object(self, object_name):
        # Remove previous object
        self._reset_robot()

        trigger_db = False
        if self.obj_id is not None:
            print('self.obj_id (before clearning) = ', self.obj_id)
            print('n bodies (before clearning) = ', self.p.getNumBodies())
            self.p.removeBody(self.obj_id)
            print('self.obj_id (after clearning) = ', self.obj_id)
            print('n bodies (after clearning) = ', self.p.getNumBodies())

            trigger_db = True
            pdb.set_trace()

        # Set a new one
        if trigger_db:
            print('self.obj_id (before adding) = ', self.obj_id)

        self._init_object(
            object_name=object_name, object_position=self.object_position, object_xyzw=self.object_xyzw
        )

        if trigger_db:
            print('self.obj_id (after adding) = ', self.obj_id)

        if trigger_db:
            pdb.set_trace()

        self._init_default_infos()
        self._run_stabilization_steps()

        self._initial_stabilized_object_pos = self.initial_stabilized_object_pos()

        # Update sim env local save
        self._init_local_sim_save()

        if trigger_db:
            pdb.set_trace()
            pass

    def initial_stabilized_object_pos(self):
        if self.obj_id is None:
            return None

        pos, qua = self.p.getBasePositionAndOrientation(self.obj_id)
        return pos, qua

    def save_rolling_state_locally(self, i_step):
        Path("tmp").mkdir(parents=True, exist_ok=True)
        rolling_state_p_file_root = "tmp/rolling_save_state"
        local_pid_str = str(os.getpid())
        rolling_state_p_local_pid_file = rolling_state_p_file_root + '_' + local_pid_str + '_' + str(i_step) + '.bullet'
        prev_rolling_state_p_local_pid_file = rolling_state_p_file_root + '_' + local_pid_str + '_' + str(self.i_step_last_rolling_step) + '.bullet'

        if os.path.exists(prev_rolling_state_p_local_pid_file):
            os.remove(prev_rolling_state_p_local_pid_file)

        self.p.saveBullet(rolling_state_p_local_pid_file)
        self.i_step_last_rolling_step = i_step


    def restore_local_rolling_state(self):

        rolling_state_p_file_root = "tmp/rolling_save_state"
        local_pid_str = str(os.getpid())
        rolling_state_p_local_pid_file = rolling_state_p_file_root + '_' + local_pid_str + '_' + str(self.i_step_last_rolling_step) + '.bullet'
        file2load = rolling_state_p_local_pid_file
        self.p.restoreState(fileName=file2load)


    def get_state(self) -> Dict[str, ArrayLike]:
        """Returns the unnormalized object position and joint positions.
        The returned dict can be passed to initialize another environment."""

        pos, qua = self.p.getBasePositionAndOrientation(self.obj_id) if self.obj_id is not None else (None, None)
        joint_positions = [s[0] for s in self.p.getJointStates(self.robot_id, self.joint_ids)]
        env_state = {'object_position': pos, 'object_xyzw': qua, 'joint_positions': joint_positions}
        return env_state

    def _apply_obj_pos_offset(self, pos, init_pos_offset):
        assert len(init_pos_offset) == 2
        pos = list(pos)
        pos[0] += init_pos_offset[0]  # x offset
        pos[1] += init_pos_offset[1]  # y offset
        return tuple(pos)

    def reset(self, delta_yaw=0., multiply_friction={},
              object_position=None, object_xyzw=None, joint_positions=None, load=None, skip_state_reload=False,
              run_name=None, force_state_load=False, init_pos_offset=None, do_noise_joints_pos=False, **kwargs):

        self._do_noise_joints_pose = do_noise_joints_pos
        self.reward_cumulated = 0
        load = load or 'state'
        load = load.strip().lower()

        assert load == 'state'
        assert self.init_state_p_file_root, 'bullet tmp file not properly set'

        if not skip_state_reload:
            if force_state_load:
                raise NotImplementedError

            else:
                local_pid_str = str(os.getpid())
                init_state_p_local_pid_file = self.init_state_p_file_root + '_' + local_pid_str
                file2load = init_state_p_local_pid_file

            self.p.restoreState(fileName=file2load)

        if delta_yaw == 0 and len(multiply_friction) == 0 and \
                object_position is None and object_xyzw is None and joint_positions is None \
                    and init_pos_offset is None:

            if skip_state_reload:
                self._reset_robot()

            return self.get_obs()  # do not need to change the position

        pos, qua = self.p.getBasePositionAndOrientation(self.obj_id)

        _, qua = self.p.multiplyTransforms([0,0,0], [0, 0, np.sin(delta_yaw/2), np.cos(delta_yaw/2)], [0, 0, 0], qua) # apply delta_yaw rotation
        pos = object_position or pos  # overwrite if absolute position is given
        qua = object_xyzw or qua
        joint_pos = joint_positions or [s[0] for s in self.p.getJointStates(self.robot_id, self.joint_ids)]

        if init_pos_offset is not None:
            pos = self._apply_obj_pos_offset(pos, init_pos_offset)

        self.p.resetBasePositionAndOrientation(self.obj_id, pos, qua)
        for id, jp in zip(self.joint_ids, joint_pos): # reset the robot
            self.p.resetJointState(self.robot_id, jointIndex=id, targetValue=jp)

        new_friction = {}
        for key, value in multiply_friction.items():
            assert key in {"lateral", "rolling", "spinning"}, f"you gave {key}, allowed keys are lateral, rolling, spinning"
            new_friction[key + 'Friction'] = value * self.frictions[key]
        self.p.changeDynamics(bodyUniqueId=self.obj_id, linkIndex=-1, **new_friction)  # set the object friction

        if skip_state_reload:
            self._reset_robot()

        return self.get_obs()

    def get_obs(self) -> ArrayLike:

        is_obj_initialized = self.obj_id is not None

        if is_obj_initialized:
            obj_pose = self.p.getBasePositionAndOrientation(self.obj_id)
            # we do not normalize the velocity, supposing the object is not moving that fast
            # we do not express the velocity in the robot frame, supoosing the robot is not moving
            obj_vel = self.p.getBaseVelocity(self.obj_id)
            self.info['object position'], self.info['object xyzw'] = obj_pose
            self.info['object linear velocity'], self.info['object angular velocity'] = obj_vel

        jointStates = self.p.getJointStates(self.robot_id, self.joint_ids)
        pos, vel = [0]*self.n_joints, [0]*self.n_joints
        for i, state, u, l, v in zip(range(self.n_joints), jointStates, self.upper_limits, self.lower_limits, self.max_velocity):
            pos[i] = 2*(state[0]-u)/(u-l) + 1  # set between -1 and 1
            vel[i] = state[1]/v  # set between -1 and 1
            self.info['joint positions'][i], self.info['joint velocities'][i], jointReactionForces, self.info['applied joint motor torques'][i] = state
            self.info['joint reaction forces'][i] = jointReactionForces[-1] # get Mz

        sensor_torques = self.info['joint reaction forces'] / self.max_force  # scale to [-1,1]
        absolute_center = self.p.multiplyTransforms(*self.p.getBasePositionAndOrientation(self.robot_id),*self.center_workspace_robot_frame)  # the pose of center_workspace in the world
        invert = self.p.invertTransform(*absolute_center)

        if is_obj_initialized:
            obj_pos, obj_or = self.p.multiplyTransforms(*invert, *obj_pose) # the object pose in the center_workspace frame

            obj_pos = np.array(obj_pos)/self.ws_radius
            obj_or = self.p.getMatrixFromQuaternion(obj_or)[:6] # taking 6 parameters from the rotation matrix to let the rotation be described in a continuous representation, which is better for neural networks

        # get information on gripper
        self.info['end effector position'], self.info['end effector xyzw'], _, _, _, _, self.info['end effector linear velocity'], self.info['end effector angular velocity'] = self.p.getLinkState(self.robot_id, self.end_effector_id, computeLinkVelocity=True)

        end_pos, end_or = self.p.multiplyTransforms(*invert, self.info['end effector position'], self.info['end effector xyzw'])
        end_pos = np.array(end_pos) / self.ws_radius
        end_or = self.p.getMatrixFromQuaternion(end_or)[:6]
        end_lin_vel, _ = self.p.multiplyTransforms(*self.p.invertTransform((0,0,0), absolute_center[1]), self.info['end effector linear velocity'], (0,0,0,1))

        self.info['robot state'] = np.hstack([end_pos, end_or, end_lin_vel, pos, vel, sensor_torques,]) # robot state without the object state

        if is_obj_initialized:
            observation = np.hstack((self.info['robot state'], obj_or, *obj_vel, obj_pos))
            assert np.isfinite(observation).all(), f"observation is not valid: {observation}"
        else:
            observation = consts.BULLET_DUMMY_OBS

        return observation

    def _reset_robot(self):
        pass  # Supposed to be overloaded in the inherited class

    def get_fingers(self, x):
        """Return the value of the fingers to control all finger with -1≤x≤1.
        Gripper opened: x=1, gripper closed: x=-1"""
        # Supposed to be overloaded in the inherited class
        raise NotImplementedError(f'get_fingers() is not implemented in {self.__name__}.')

    def render(self, mode='human'):
        if mode in {'rgb_array', 'rgba_array'}: # slow !
            camera = {**self.camera}
            if mode == 'rgb_array': # if rgb, use low resolution
                camera['height'], camera['width'] = 256, 256
            img = self.p.getCameraImage(
                width=camera['width'],
                height=camera['height'],
                viewMatrix=camera['viewMatrix'],
                projectionMatrix=camera['projectionMatrix'],
                renderer=camera['renderer'],
            )[2]
            img = np.array(img, dtype=np.uint8).reshape(camera['height'], camera['width'], 4)
            return img[:,:,:3] if mode=='rgb_array' else img
        elif mode == 'human':
            pass
        else:
            super().render(mode=mode) # just raise an exception

    def close(self):
        if self.physicsClientId >=0:
            self.p.disconnect()
            self.physicsClientId = -1

    def get_end_effector_state(self, normalized=True):
        end_effector_link_state = self.p.getLinkState(self.robot_id, self.end_effector_id)
        end_eff_xyz = end_effector_link_state[0]

        end_eff_xyzw = end_effector_link_state[1]
        end_eff_euler = self.p.getEulerFromQuaternion(end_eff_xyzw)

        end_effector_6dof_pos = np.concatenate([end_eff_xyz, end_eff_euler], axis=0)
        return end_effector_6dof_pos

    def get_joint_state(self, position=True, normalized=True):
        """ Return (un)normalized joint positions (velocities) without the gripper"""
        #pdb.set_trace()
        if position:
            js_upper_lim = self.upper_limits[self.n_body_gripper:-self.n_control_gripper]
            js_lower_lim = self.lower_limits[self.n_body_gripper:-self.n_control_gripper]
            i_state = 0
            joint_state = np.array(
                [s[i_state] for s in self.p.getJointStates(
                    self.robot_id, self.joint_ids[self.n_body_gripper:-self.n_control_gripper]
                )]
            )

            if normalized:
                joint_state = 2 * (joint_state - js_upper_lim) / (js_upper_lim - js_lower_lim) + 1

            return joint_state
        else:
            i_vel = 1
            joint_vel = np.array(
                [js[i_vel] for js in self.p.getJointStates(
                    self.robot_id, self.joint_ids[self.n_body_gripper:-self.n_control_gripper]
                )]
            )
            if normalized:
                joint_vel = joint_vel / self.max_velocity[self.n_body_gripper:-self.n_control_gripper]
            return joint_vel

    def _run_stabilization_steps(self, skip_robot_reset=False):
        print('n_stabilization_steps=', consts.N_ITER_STABILIZING_SIM_DEFAULT_ROBOT_GRASP)
        self._update_n_step_sim(n_steps=consts.N_ITER_STABILIZING_SIM_DEFAULT_ROBOT_GRASP)

        if not skip_robot_reset:
            # No command is given to the robot during stabiliation steps => robot state is likely to change, which
            # should be avoided.
            self._reset_robot()

    def _update_n_step_sim(self, n_steps):
        for _ in range(n_steps):
            self.p.stepSimulation()
            #time.sleep(0.01)

    def get_joints_poses_from_ik(self, pos, or_pry, normalized=False):

        if isinstance(or_pry, np.ndarray):
            or_pry = or_pry.tolist()
        # pdb.set_trace()

        assert isinstance(or_pry, list) and len(or_pry) == 3
        orn = p.getQuaternionFromEuler(or_pry)

        rp = self._get_rest_poses()
        arm_controllable_joint_ids = self._get_arm_controllable_joint_ids()

        ll = [
            self.p.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=j_id)[8]
            for j_id in arm_controllable_joint_ids
        ]
        ul = [
            self.p.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=j_id)[9]
            for j_id in arm_controllable_joint_ids
        ]
        jr = [5.8] * len(arm_controllable_joint_ids) # based on pybullet official repo example

        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_id,
            pos,
            orn,
            lowerLimits=ll,
            upperLimits=ul,
            jointRanges=jr,
            restPoses=rp
        )

        controllable_joint_ids = self._get_controllable_joint_ids()

        try:
            assert len(joint_poses) == len(controllable_joint_ids)
        except:
            pdb.set_trace()

        n_arm_jp = len(self._get_arm_controllable_joint_ids())
        arm_joint_poses = np.array(joint_poses[:n_arm_jp])

        if normalized:
            js_upper_lim = self.upper_limits[self.n_body_gripper:-self.n_control_gripper]
            js_lower_lim = self.lower_limits[self.n_body_gripper:-self.n_control_gripper]

            arm_joint_poses = 2 * (arm_joint_poses - js_upper_lim) / (js_upper_lim - js_lower_lim) + 1

        return arm_joint_poses

    def apply_cartesian_ik_pose(self, pos, or_pry=None, n_steps=None, display=False, accurate=False, plot_target=False):
        if plot_target:
            robot_pos = self.p.getBasePositionAndOrientation(self.robot_id)
            origin = tuple(pos), robot_pos[1]
            print('displaying target : ', origin)
            self._debug_plot_shadow_object_at_pos(pos=origin)

        is_ik_accurate = accurate
        null_space_flg = True
        is_ik_pos_or_pry = or_pry is not None and not null_space_flg
        is_ik_null_space = or_pry is not None and null_space_flg

        if is_ik_accurate:
            joint_poses = self.accurateIK(
                bodyId=self.robot_id, end_effector_id=self.end_effector_id, targetPosition=pos,
                targetOrientation=[0., 0., 0., 0.], lowerLimits=self.lower_limits, upperLimits=self.upper_limits,
                joint_ranges=self.joint_ranges, rest_poses=self.rest_poses, useNullSpace=False
            )

        elif is_ik_pos_or_pry:
            assert isinstance(or_pry, list) and len(or_pry) == 3
            orn = p.getQuaternionFromEuler(or_pry)
            joint_damping_gain = [0.00001] if self._is_baxter() else [0.01]

            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_id,
                pos,
                orn,
                jointDamping=joint_damping_gain * self.n_controllable_joints
            )
            controllable_joint_ids = self._get_controllable_joint_ids()
            assert len(joint_poses) == len(controllable_joint_ids)

        elif is_ik_null_space:
            if isinstance(or_pry, np.ndarray):
                or_pry = or_pry.tolist()

            assert isinstance(or_pry, list) and len(or_pry) == 3
            orn = p.getQuaternionFromEuler(or_pry)
            rp = self._get_rest_poses()
            arm_controllable_joint_ids = self._get_arm_controllable_joint_ids()

            ll = [
                self.p.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=j_id)[8]
                for j_id in arm_controllable_joint_ids
            ]
            ul = [
                self.p.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=j_id)[9]
                for j_id in arm_controllable_joint_ids
            ]
            jr = [5.8] * len(arm_controllable_joint_ids)  # based on pybullet official repo example


            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_id,
                pos,
                orn,
                lowerLimits=ll,
                upperLimits=ul,
                jointRanges=jr,
                restPoses=rp
            )
            pdb.set_trace()
            controllable_joint_ids = self._get_controllable_joint_ids()
            assert len(joint_poses) == len(controllable_joint_ids)

        else:
            joint_damping_gain = [0.00001] if self._is_baxter() else [0.01]
            joint_poses = self.p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=self.end_effector_id, #25, #self.end_effector_id,
                targetPosition=pos,
                jointDamping=joint_damping_gain * self.n_controllable_joints
            )
            # Note : len(joint_poses) == len(controllable_joints) (/!\ and not the CONTROLLED ones)
            controllable_joint_ids = self._get_controllable_joint_ids()
            assert len(joint_poses) == len(controllable_joint_ids)

        for j_id, j_pose in zip(controllable_joint_ids, joint_poses):
            self.p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=j_id,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=j_pose,
                targetVelocity=0,
                force=500,
                positionGain=0.03,
                velocityGain=1
            )

        if n_steps is not None:
            assert isinstance(n_steps, int)
            assert n_steps > 0

            if display:
                assert self.display

                self._update_n_step_sim_display_debug(n_steps=n_steps, joint_poses=joint_poses)


            else:
                self._update_n_step_sim(n_steps=n_steps)

    def _get_rest_poses(self):
        raise NotImplementedError('Must be overloaded in robot subclasses.')

    def debug_plot_6dof_pos(self, pos, or_pry, color=[1, 1, 0, 1], shadow_obj_dim=0.01):
        or_xyzw = self.p.getQuaternionFromEuler(or_pry)
        self._debug_plot_shadow_object_at_pos(pos=(pos, or_xyzw), color=color, shadow_obj_dim=shadow_obj_dim)

    def _debug_plot_shadow_object_at_pos(self, pos, color=[1, 1, 0, 1], shadow_obj_dim=0.01):
        info_shape_so_vis = {"shapeType": self.p.GEOM_BOX,
                             "halfExtents": [shadow_obj_dim / 2, shadow_obj_dim / 2, shadow_obj_dim / 2],
                             "visualFramePosition": pos[0]}
        info_shape_so_col = {"shapeType": self.p.GEOM_BOX,
                             "halfExtents": [shadow_obj_dim / 2, shadow_obj_dim / 2, shadow_obj_dim / 2],
                             "collisionFramePosition": pos[0]}
        shadow_obj_id = self._debug_generate_obj_id(info_shape_so_vis, info_shape_so_col, color=color)

    def _debug_plot_end_effector_pos(self):
        SHADOW_OBJ_DIM = 0.11  #0.01 #1.11
        info_shape_so = {"shapeType": self.p.GEOM_BOX,
                         "halfExtents": [SHADOW_OBJ_DIM / 2, SHADOW_OBJ_DIM / 2, SHADOW_OBJ_DIM / 2],
        }
        shadow_obj_id = self._debug_generate_obj_id(info_shape_so, color=[1, 0, 1, 1])

        pdb.set_trace()
        self.p.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.end_effector_id)
        pos_so, init_obj_xyzw = self.p.getBasePositionAndOrientation(self.end_effector_id)
        self.p.resetBasePositionAndOrientation(shadow_obj_id, pos_so, init_obj_xyzw)

    def _debug_generate_obj_id(self, info_shape_so_vis, info_shape_so_col, color, no_collision=True):
        base_vis_shape = self.p.createVisualShape(**info_shape_so_vis, rgbaColor=color)
        base_collision_shape = self.p.createCollisionShape(**info_shape_so_col)
        if no_collision:
            debug_obj_id = self.p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=base_vis_shape,
                useMaximalCoordinates=False
            )
        else:
            debug_obj_id = self.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=base_collision_shape,
                baseVisualShapeIndex=base_vis_shape,
                useMaximalCoordinates=False
            )

        self.debug_i_debug_bodies.append(debug_obj_id)
        return debug_obj_id

    def debug_remove_debug_bodies(self):
        for body_id in self.debug_i_debug_bodies:
            self.p.removeBody(body_id)
            print(f'body_id={body_id} removed')

    def accurateIK(self, bodyId, end_effector_id, targetPosition, targetOrientation, lowerLimits, upperLimits,
                   joint_ranges, rest_poses, useNullSpace=False, maxIter=10, threshold=1e-4):

        if useNullSpace:
            jointPoses = self.p.calculateInverseKinematics(bodyId, end_effector_id, targetPosition,
                                                           targetOrientation=targetOrientation,
                                                           lowerLimits=lowerLimits, upperLimits=upperLimits,
                                                           jointRanges=joint_ranges,
                                                           restPoses=rest_poses)
        else:
            jointPoses = self.p.calculateInverseKinematics(bodyId, end_effector_id, targetPosition,
                                                           targetOrientation=targetOrientation)

        return jointPoses

    def _update_n_step_sim_display(self, n_steps):
        for i_step in range(n_steps):
            self._update_n_step_sim(n_steps=1)
            time.sleep(0.01)

    def _update_n_step_sim_display_debug(self, n_steps, joint_poses=None):
        for i_step in range(n_steps):
            self._update_n_step_sim(n_steps=1)

            curr_j_pose = tuple(self.get_state()['joint_positions'])
            verbose_str = f'target = {joint_poses} \ncurrent = {curr_j_pose}\n' + '-' * 50
            verbose_str = None

            if verbose_str is not None:
                print(verbose_str)

            time.sleep(0.001)



