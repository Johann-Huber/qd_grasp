
import gym_envs.envs.src.robots.kuka_based.kuka_consts as ku_consts

# ---------------------------------------------- #
#                  JOINT IDS
# ---------------------------------------------- #


KUKA_ARM_JOINT_ID_STATUS = {
    0:  {'name': 'J0',          'status': 'CONTROLLED',         'is_controllable': True},
    1:  {'name': 'J1',          'status': 'CONTROLLED',         'is_controllable': True},
    2:  {'name': 'J2',          'status': 'CONTROLLED',         'is_controllable': True},
    3:  {'name': 'J3',          'status': 'CONTROLLED',         'is_controllable': True},
    4:  {'name': 'J4',          'status': 'CONTROLLED',         'is_controllable': True},
    5:  {'name': 'J5',          'status': 'CONTROLLED',         'is_controllable': True},
    6:  {'name': 'J6',          'status': 'CONTROLLED',         'is_controllable': True},
}

KUKA_CLAW_GRIP_JOINT_ID_STATUS = {
    7 :   {'name': 'gripper_to_arm',           'status': 'FIXED',        'part': 'gripper',   'is_controllable': False},
    8 :   {'name': 'base_left_finger_joint',   'status': 'CONTROLLED',   'part': 'gripper',   'is_controllable': True},
    9 :   {'name': 'left_finger_base_joint',   'status': 'FIXED',        'part': 'gripper',   'is_controllable': False},
    10:   {'name': 'left_base_tip_joint',      'status': 'CONTROLLED',   'part': 'gripper',   'is_controllable': True},
    11:   {'name': 'base_right_finger_joint',  'status': 'CONTROLLED',   'part': 'gripper',   'is_controllable': True},
    12:   {'name': 'right_finger_base_joint',  'status': 'FIXED',        'part': 'gripper',   'is_controllable': False},
    13:   {'name': 'right_base_tip_joint',     'status': 'CONTROLLED',   'part': 'gripper',   'is_controllable': True},
    14:   {'name': 'end_effector_joint',       'status': 'FIXED',        'part': 'gripper',   'is_controllable': False},
}



# ---------------------------------------------- #
#                 HYPERPARAMETERS
# ---------------------------------------------- #

KUKA_WS_RADIUS = 1.2


KUKA_DEFAULT_INIT_OBJECT_POSITION = [0., 0.15, -0.18318535463122008]

KUKA_JOINT_IDS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 13]

KUKA_CONTACT_IDS = [8, 9, 10, 11, 12, 13]


KUKA_N_CONTROL_GRIPPER = 4
KUKA_END_EFFECTOR_ID = 6
KUKA_CENTER_WORKSPACE = 0
KUKA_DISABLED_COLLISION_PAIRS = [[8, 11], [10, 13]]



KUKA_ABOVE_OBJECT_INIT_POSITION_GRIPPER = [
    -0.04999999999999999, -0.00105972150624707, -0.014672116051631068, 0.016698095337863785
]
KUKA_ABOVE_OBJECT_INIT_POSITION = \
    ku_consts.KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS + KUKA_ABOVE_OBJECT_INIT_POSITION_GRIPPER


KUKA_INIT_POS_X_MIN, KUKA_INIT_POS_X_MAX = -0.5, 0.5
KUKA_INIT_POS_Y_MIN, KUKA_INIT_POS_Y_MAX = 0, 0.5
KUKA_INIT_POS_Z_MIN, KUKA_INIT_POS_Z_MAX = -0.3, 0.2