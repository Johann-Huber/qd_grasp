

# ---------------------------------------------- #
#                  JOINT IDS
# ---------------------------------------------- #

# Definition

FRANKA_EMIKA_PANDA_ARM_JOINT_ID_STATUS = {
    0: {'name': 'panda_joint1', 'status': 'CONTROLLED', 'part': 'fixed_wrist', 'is_controllable': True},
    1: {'name': 'panda_joint2', 'status': 'CONTROLLED', 'part': 'fixed_wrist', 'is_controllable': True},
    2: {'name': 'panda_joint3', 'status': 'CONTROLLED', 'part': 'index_finger', 'is_controllable': True},
    3: {'name': 'panda_joint4', 'status': 'CONTROLLED', 'part': 'index_finger', 'is_controllable': True},
    4: {'name': 'panda_joint5', 'status': 'CONTROLLED', 'part': 'mid_finger', 'is_controllable': True},
    5: {'name': 'panda_joint6', 'status': 'CONTROLLED', 'part': 'mid_finger', 'is_controllable': True},
    6: {'name': 'panda_joint7', 'status': 'CONTROLLED', 'part': 'ring_finger', 'is_controllable': True},
    7: {'name': 'panda_joint8', 'status': 'FIXED', 'part': 'ring_finger', 'is_controllable': False},
}

PANDA_2_FINGERS_GRIP_JOINT_ID_STATUS = {
    8: {'name': 'panda_hand_joint', 'status': 'FIXED', 'part': 'last_finger', 'is_controllable': False},
    9: {'name': 'panda_hand_tcp_joint', 'status': 'FIXED', 'part': 'last_finger', 'is_controllable': False},
    10: {'name': 'panda_finger_joint1', 'status': 'CONTROLLED', 'part': 'fixed_wrist', 'is_controllable': True},
    11: {'name': 'panda_finger_joint2', 'status': 'CONTROLLED', 'part': 'fixed_wrist', 'is_controllable': True},
}



# Key variables

# --- General

FRANKA_EMIKA_PANDA_ARM_ALL_JOINT_IDS = [j_id for j_id in FRANKA_EMIKA_PANDA_ARM_JOINT_ID_STATUS]
PANDA_2_FINGERS_GRIP_ALL_JOINT_IDS = [j_id for j_id in PANDA_2_FINGERS_GRIP_JOINT_ID_STATUS]

# --- Controlled

FRANKA_EMIKA_PANDA_ARM_CONTROLLED_JOINT_IDS = [
    j_id for j_id in FRANKA_EMIKA_PANDA_ARM_JOINT_ID_STATUS
    if FRANKA_EMIKA_PANDA_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
]

PANDA_2_FINGERS_GRIP_CONTROLLED_JOINT_IDS = [
    j_id for j_id in PANDA_2_FINGERS_GRIP_JOINT_ID_STATUS
    if PANDA_2_FINGERS_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
]

# --- Controllable

FRANKA_EMIKA_PANDA_ARM_CONTROLLABLE_JOINT_IDS = [
    j_id for j_id in FRANKA_EMIKA_PANDA_ARM_JOINT_ID_STATUS
    if FRANKA_EMIKA_PANDA_ARM_JOINT_ID_STATUS[j_id]['is_controllable']
]

PANDA_2_FINGERS_GRIP_CONTROLLABLE_JOINT_IDS = [
    j_id for j_id in PANDA_2_FINGERS_GRIP_JOINT_ID_STATUS
    if PANDA_2_FINGERS_GRIP_JOINT_ID_STATUS[j_id]['is_controllable']
]



# ---------------------------------------------- #
#                 INITIAL STATE
# ---------------------------------------------- #

# Base

#KUKA_ALLEGRO_BASE_POSITION = [0, -0.5, -0.5]
PANDA_2_FINGERS_BASE_ORIENTATION = [0., 0., 0., 1.]


# Default state
ARM_DEFAULT_JOINT_STATES = {
    0: -0.16,
    1: -0.79,
    2: 0.14,
    3: -2.34,
    4: 0.0957,
    5: 1.5664,
    6: 0.6692
}

HAND_DEFAULT_JOINT_STATES = {
    10: 0.04,
    11: 0.04
}

DEFAULT_JOINT_STATES = {**ARM_DEFAULT_JOINT_STATES, **HAND_DEFAULT_JOINT_STATES}






# ---------------------------------------------- #
#                    3D Models
# ---------------------------------------------- #

PANDA_2_FINGERS_GRIP_RELATIVE_PATH_XACRO = 'franka_description/robots/panda/panda.urdf.xacro'
PANDA_2_FINGERS_GRIP_RELATIVE_PATH_GENERATED_URDF = 'generated/franka_emika_panda.urdf'

# ---------------------------------------------- #
#                    SCENE
# ---------------------------------------------- #


FRANKA_END_EFFECTOR_JOINT_ID = 8



# TODO must be verified carefully -----------------

FRANKA_INIT_POS_X_MIN, FRANKA_INIT_POS_X_MAX = -0.20, 0.15  #-0.1, 0.4
FRANKA_INIT_POS_Y_MIN, FRANKA_INIT_POS_Y_MAX = 0., 0.30  #-0.1, 0.5
FRANKA_INIT_POS_Z_MIN, FRANKA_INIT_POS_Z_MAX = -0.20, 0.  #-0.1, 0.2

FRANKA_DEFAULT_INIT_OBJECT_POSITION = [0.05, 0.15, -0.20] #[0., 0.15, -0.18318535463122008]

EUROBIN_FRANKA_DEFAULT_INIT_OBJECT_POSITION = [0.25, 0.15, -0.20]

EUROBIN_Y_OFFSET = 0.25
EUROBIN_X_OFFSET = 0.1


FRANKA_EMIKA_PANDA_INIT_POSITION_JOINTS = [
    -0.16, -0.79, 0.14, -2.34, 0.0957, 1.5664, 0.6692
]

# TODO must be verified carefully /-----------------


PANDA_2_FINGERS_CENTER_WORKSPACE = 0
PANDA_2_FINGERS_WS_RADIUS = 1
PANDA_2_FINGERS_DISABLE_COLLISION_PAIRS = []
PANDA_2_FINGERS_ALLOWED_COLLISION_PAIRS = []


