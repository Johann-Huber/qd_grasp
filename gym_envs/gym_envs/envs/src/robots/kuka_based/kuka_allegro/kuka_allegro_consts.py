
import numpy as np
import gym_envs.envs.src.robots.kuka_based.kuka_consts as ku_consts

# ---------------------------------------------- #
#                  JOINT IDS
# ---------------------------------------------- #

# Definition

KUKA_ARM_JOINT_ID_STATUS = {
    0:  {'name': 'J0',          'status': 'CONTROLLED',         'is_controllable': True},
    1:  {'name': 'J1',          'status': 'CONTROLLED',         'is_controllable': True},
    2:  {'name': 'J2',          'status': 'CONTROLLED',         'is_controllable': True},
    3:  {'name': 'J3',          'status': 'CONTROLLED',         'is_controllable': True},
    4:  {'name': 'J4',          'status': 'CONTROLLED',         'is_controllable': True},
    5:  {'name': 'J5',          'status': 'CONTROLLED',         'is_controllable': True},
    6:  {'name': 'J6',          'status': 'CONTROLLED',         'is_controllable': True},
}

KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS = {
    9:  {'name': 'joint_0',           'status': 'FIXED',        'part': 'index_finger',   'is_controllable': True},
    10: {'name': 'joint_1',           'status': 'CONTROLLED',   'part': 'index_finger',   'is_controllable': True},
    11: {'name': 'joint_2',           'status': 'CONTROLLED',   'part': 'index_finger',   'is_controllable': True},
    12: {'name': 'joint_3',           'status': 'CONTROLLED',   'part': 'index_finger',   'is_controllable': True},
    13: {'name': 'joint_3_tip',       'status': 'FIXED',        'part': 'index_finger',   'is_controllable': False},

    14: {'name': 'joint_4',           'status': 'FIXED',        'part': 'mid_finger',   'is_controllable': True},
    15: {'name': 'joint_5',           'status': 'CONTROLLED',   'part': 'mid_finger',   'is_controllable': True},
    16: {'name': 'joint_6',           'status': 'CONTROLLED',   'part': 'mid_finger',   'is_controllable': True},
    17: {'name': 'joint_7',           'status': 'CONTROLLED',   'part': 'mid_finger',   'is_controllable': True},
    18: {'name': 'joint_7_tip',       'status': 'FIXED',        'part': 'mid_finger',   'is_controllable': False},

    19: {'name': 'joint_8',           'status': 'FIXED',        'part': 'last_finger',   'is_controllable': True},
    20: {'name': 'joint_9',           'status': 'CONTROLLED',   'part': 'last_finger',   'is_controllable': True},
    21: {'name': 'joint_10',          'status': 'CONTROLLED',   'part': 'last_finger',   'is_controllable': True},
    22: {'name': 'joint_11',          'status': 'CONTROLLED',   'part': 'last_finger',   'is_controllable': True},
    23: {'name': 'joint_11_tip',      'status': 'FIXED',        'part': 'last_finger',   'is_controllable': False},

    24: {'name': 'joint_12',          'status': 'FIXED',        'part': 'thumb',   'is_controllable': True},
    25: {'name': 'joint_13',          'status': 'FIXED',        'part': 'thumb',   'is_controllable': True},
    26: {'name': 'joint_14',          'status': 'CONTROLLED',   'part': 'thumb',   'is_controllable': True},
    27: {'name': 'joint_15',          'status': 'CONTROLLED',   'part': 'thumb',   'is_controllable': True},
}

# Key variables

# --- General

KUKA_ALLEGRO_GRIP_ALL_JOINT_IDS = [j_id for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS]

# --- Controlled

KUKA_ARM_CONTROLLED_JOINT_IDS = [
    j_id for j_id in KUKA_ARM_JOINT_ID_STATUS
    if KUKA_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
]

ALLEGRO_HAND_CONTROLLED_JOINT_IDS = [
    j_id for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
    if KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
]

# --- Controllable

KUKA_ARM_CONTROLLABLE_JOINT_IDS = [
    j_id for j_id in KUKA_ARM_JOINT_ID_STATUS
    if KUKA_ARM_JOINT_ID_STATUS[j_id]['is_controllable']
]

ALLEGRO_HAND_CONTROLLABLE_JOINT_IDS = [
    j_id for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
    if KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['is_controllable']
]



# ---------------------------------------------- #
#                 HYPERPARAMETERS
# ---------------------------------------------- #

# Hand closing primitive utils
DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_THUMB = [15, 16, 17, 20, 21, 22]

DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_THUMB = [10, 11, 12, 20, 21, 22]

DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_MID_THUMB = [20, 21, 22]
DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_LAST_THUMB = [10, 11, 12]

DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX = [10, 11]
DISCARD_NO_JOINT_IDS = []

DISCARDED_J_IDS_CLOSING_PRIMITIVES = {
    0: DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_THUMB,
    1: DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_THUMB,
    2: DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_MID_THUMB,
    3: DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_MID_THUMB,
    4: DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_LAST_THUMB,
    5: DISCARD_NO_JOINT_IDS,
    6: DISCARD_NO_JOINT_IDS
}


KUKA_ALLEGRO_JOINT_IDS = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27]

KUKA_ALLEGRO_INIT_POS_X_MIN, KUKA_ALLEGRO_INIT_POS_X_MAX = -0.3, 0.3
KUKA_ALLEGRO_INIT_POS_Y_MIN, KUKA_ALLEGRO_INIT_POS_Y_MAX = -0.1, 0.3
KUKA_ALLEGRO_INIT_POS_Z_MIN, KUKA_ALLEGRO_INIT_POS_Z_MAX = -1.0, -0.5


N_CONTROLLABLE_ALLEGRO = np.array([
    int(KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['is_controllable']) for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
    ]).sum()
ALLEGRO_ABOVE_OBJECT_INIT_POSITION_GRIPPER = [0] * N_CONTROLLABLE_ALLEGRO

KUKA_ABOVE_OBJECT_INIT_POSITION = \
    ku_consts.KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS + ALLEGRO_ABOVE_OBJECT_INIT_POSITION_GRIPPER


KUKA_ALLEGRO_RELATIVE_PATH_XACRO = 'lbr_iiwa/urdf/lbr_iiwa_14_r820_allegro.xacro'
KUKA_ALLEGRO_RELATIVE_PATH_GENERATED_URDF = 'generated/kuka_iiwa_allegro_right.urdf'


# ---------------------------------------------- #
#                 INITIAL STATE
# ---------------------------------------------- #

# Base

KUKA_ALLEGRO_BASE_POSITION = [0, -0.5, -0.5]
KUKA_ALLEGRO_BASE_ORIENTATION = [0., 0., 0., 1.]


# Default state
assert len(KUKA_ALLEGRO_JOINT_IDS) == len(KUKA_ABOVE_OBJECT_INIT_POSITION)
DEFAULT_JOINT_STATES = {j_id: j_val for j_id, j_val in zip(KUKA_ALLEGRO_JOINT_IDS, KUKA_ABOVE_OBJECT_INIT_POSITION)}


# Manually fixed states

J_ID_THUMB_OPPOSITION = 24
OPPOSED_THUMB_INIT_VALUE = 1.396

J_ID_WRIST = 6
PALM_FACING_TABLE_INIT_VALUE = np.pi

ARM_MANUALLY_PRESET_VALUES = {
    J_ID_WRIST: PALM_FACING_TABLE_INIT_VALUE,
}
HAND_MANUALLY_PRESET_VALUES = {
    J_ID_THUMB_OPPOSITION: OPPOSED_THUMB_INIT_VALUE,
}

MANUALLY_PRESET_VALUES = {**ARM_MANUALLY_PRESET_VALUES, **HAND_MANUALLY_PRESET_VALUES}


# ---------------------------------------------- #
#                   COMMANDS
# ---------------------------------------------- #

OPEN_GRIP_COMMAND_VALUE = 1.
CLOSE_GRIP_COMMAND_VALUE = -1.
VALID_GRIP_COMMAND_VALUES = [OPEN_GRIP_COMMAND_VALUE, CLOSE_GRIP_COMMAND_VALUE]



# ---------------------------------------------- #
#                    SCENE
# ---------------------------------------------- #

KUKA_ALLEGRO_DEFAULT_OBJECT_POSE = [0, 0.1, 0]

KUKA_ALLEGRO_CENTER_WORKSPACE = 0

KUKA_ALLEGRO_DISABLE_COLLISION_PAIRS = []
KUKA_ALLEGRO_CHANGE_DYNAMICS = {**{}, **{j_id: {'maxJointVelocity': 0.5} for j_id in KUKA_ARM_JOINT_ID_STATUS}}

KUKA_ALLEGRO_TABLE_HEIGHT = 0.8
KUKA_ALLEGRO_END_EFFECTOR_JOINT_ID = 7


