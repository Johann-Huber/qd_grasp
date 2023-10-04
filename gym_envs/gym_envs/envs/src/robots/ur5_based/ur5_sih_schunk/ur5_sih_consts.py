
import numpy as np

# ---------------------------------------------- #
#                  JOINT IDS
# ---------------------------------------------- #

# Definition


UR5_ARM_JOINT_ID_STATUS = {
    0:  {'name': 'shoulder_pan_joint',             'status': 'CONTROLLED',         'is_controllable': True},
    1:  {'name': 'shoulder_lift_joint',            'status': 'CONTROLLED',         'is_controllable': True},
    2:  {'name': 'elbow_joint',                    'status': 'CONTROLLED',         'is_controllable': True},
    3:  {'name': 'wrist_1_joint',                  'status': 'CONTROLLED',         'is_controllable': True},
    4:  {'name': 'wrist_2_joint',                  'status': 'CONTROLLED',         'is_controllable': True},
    5:  {'name': 'wrist_3_joint',                  'status': 'CONTROLLED',         'is_controllable': True},
    6:  {'name': 'ee_fixed_joint',                 'status': 'FIXED',              'is_controllable': False},
    7:  {'name': 'wrist_3_link-tool0_fixed_joint', 'status': 'FIXED',              'is_controllable': False},
}


ALL_HAND_PARTS = ['thumb', 'index_finger', 'mid_finger', 'ring_finger', 'last_finger']


SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS = {
    8 :  {'name': 'ft_joint',                               'status': 'FIXED',              'part': 'fixed_wrist',       'is_controllable': False},
    9 :  {'name': 'schunk_sih_right_base_joint',                    'status': 'FIXED',              'part': 'fixed_wrist',       'is_controllable': False},
    10:  {'name': 'schunk_sih_right_index_finger',          'status': 'CONTROLLED',         'part': 'index_finger',      'is_controllable': True}, #CONTROLLED
    11:  {'name': 'schunk_sih_right_index_distal_joint',    'status': 'CONTROLLED',         'part': 'index_finger',      'is_controllable': True}, #CONTROLLED
    12:  {'name': 'schunk_sih_right_middle_finger',         'status': 'CONTROLLED',         'part': 'mid_finger',        'is_controllable': True}, #CONTROLLED
    13:  {'name': 'schunk_sih_right_middle_distal_joint',       'status': 'CONTROLLED',         'part': 'mid_finger',        'is_controllable': True}, #CONTROLLED
    14:  {'name': 'schunk_sih_right_ring_finger',            'status': 'CONTROLLED',         'part': 'ring_finger',       'is_controllable': True}, #CONTROLLED
    15:  {'name': 'schunk_sih_right_ring_distal_joint',       'status': 'CONTROLLED',         'part': 'ring_finger',       'is_controllable': True}, #CONTROLLED
    16:  {'name': 'schunk_sih_right_pinky_proximal_joint',            'status': 'CONTROLLED',         'part': 'last_finger',       'is_controllable': True}, #CONTROLLED
    17:  {'name': 'schunk_sih_right_pinky_distal_joint',       'status': 'CONTROLLED',         'part': 'last_finger',       'is_controllable': True}, #CONTROLLED
    18:  {'name': 'schunk_sih_right_thumb_opposition',            'status': 'FIXED',         'part': 'thumb',             'is_controllable': True},  # thumb pitch
    19:  {'name': 'schunk_sih_right_thumb_flexion',        'status': 'CONTROLLED',         'part': 'thumb',             'is_controllable': True}, #CONTROLLED
    20:  {'name': 'schunk_sih_right_thumb_distal',          'status': 'CONTROLLED',         'part': 'thumb',             'is_controllable': True}, #CONTROLLED
    21:  {'name': 'base_link-base_fixed_joint',     'status': 'FIXED',              'part': 'fixed_wrist',       'is_controllable': False},
}

# SHOULD BE FIXED AND THEREFORE NOT USED (but usable if required)
BONN_SCENE_RELATED_JOINTS = {
    22: {'name': 'table_surface_joint'},
    23: {'name': 'back_wall_joint'},
    24: {'name': 'front_wall_joint'},
    25: {'name': 'left_side_wall_joint'},
    26: {'name': 'right_side_wall_joint'},
}




# Key variables

# --- General

UR5_ARM_ALL_JOINT_IDS = [j_id for j_id in UR5_ARM_JOINT_ID_STATUS]
SIH_SCHUNK_HAND_ALL_JOINT_IDS = [j_id for j_id in SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS]

# --- Controlled

UR5_ARM_CONTROLLED_JOINT_IDS = [
    j_id for j_id in UR5_ARM_JOINT_ID_STATUS
    if UR5_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
]

SIH_SCHUNK_HAND_CONTROLLED_JOINT_IDS = [
    j_id for j_id in SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS
    if SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
]

# --- Controllable

UR5_ARM_CONTROLLABLE_JOINT_IDS = [
    j_id for j_id in UR5_ARM_JOINT_ID_STATUS
    if UR5_ARM_JOINT_ID_STATUS[j_id]['is_controllable']
]

SIH_SCHUNK_HAND_CONTROLLABLE_JOINT_IDS = [
    j_id for j_id in SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS
    if SCHUNK_SIH_RIGHT_HAND_JOINT_ID_STATUS[j_id]['is_controllable']
]



# ---------------------------------------------- #
#                 HYPERPARAMETERS
# ---------------------------------------------- #






SCHUNK_SIH_RIGHT_END_EFFECTOR_JOINT_ID = 9  # 24 #9  # to test

# Hand closing primitive utils
DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_THUMB = [12, 13, 14, 15, 16, 17]
DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_THUMB = [10, 11, 14, 15, 16, 17]
DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_MID_THUMB = [14, 15, 16, 17]
DISCARD_ALL_JOINT_IDS_BUT_NOT_RING_LAST_THUMB = [10, 11, 12, 13]
DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX = [10, 11]
#DISCARD_ALL_JOINT_IDS_BUT_NOT_MID = [12, 13] # no really useful ... a little provocative
DISCARD_NO_JOINT_IDS = []

DISCARDED_J_IDS_CLOSING_PRIMITIVES = {
    0: DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_THUMB,
    1: DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_THUMB,
    2: DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_MID_THUMB,
    3: DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_THUMB,  #DISCARD_ALL_JOINT_IDS_BUT_NOT_RING_LAST_THUMB
    4: DISCARD_NO_JOINT_IDS,
    5: DISCARD_NO_JOINT_IDS,
    6: DISCARD_NO_JOINT_IDS
}


# ---------------------------------------------- #
#                    PATHS
# ---------------------------------------------- #

UR5_SIH_SCHUNK_RELATIVE_PATH_XACRO = 'ur5_sih_schunk/ur5_sih_schunk.urdf.xacro'
UR5_SIH_SCHUNK_RELATIVE_PATH_GENERATED_URDF = 'generated/ur5_sih_schunk.urdf'


# ---------------------------------------------- #
#                 INITIAL STATE
# ---------------------------------------------- #


# Default state
ARM_DEFAULT_JOINT_STATES = {
    0: 0.90, #-0.69, #-39.5340878640268
    1: -1.75, #-76.77634454753031
    2: 2.03, #99.12169855763241
    3: -0.38, #-21.772396214971284
    4: 0.69, #39.5340878640268
    5: 1.57 #89.95437383553926
}

REST_POSE_ARM_JOINTS = [j_pose for j_id, j_pose in ARM_DEFAULT_JOINT_STATES.items()]
assert REST_POSE_ARM_JOINTS == [0.90, -1.75, 2.03, -0.38, 0.69, 1.57]

# By default set to max value
HAND_DEFAULT_JOINT_STATES = {
    8: -1.0,
    9: -1.0,
    10: 0.0,
    11: 0.0,
    12: 0.0,
    13: 0.0,
    14: 0.0,
    15: 0.0,
    16: 0.0,
    17: 0.0,
    18: 0.0,
    19: 1.571,
    20: 1.571,
    21: -1.0,
}

DEFAULT_JOINT_STATES = {**ARM_DEFAULT_JOINT_STATES, **HAND_DEFAULT_JOINT_STATES}

ARM_MANUALLY_SET_JOINT_STATES = {}

# 18: thumb pitch is set to min val to get the hand wide open
HAND_MANUALLY_SET_JOINT_STATES = {
    18: (-np.pi / 2),
    19: 0,
    20: 0,
}

MANUALLY_SET_JOINT_STATES = {**ARM_MANUALLY_SET_JOINT_STATES, **HAND_MANUALLY_SET_JOINT_STATES}



# ---------------------------------------------- #
#                 BULLET DYNAMICS
# ---------------------------------------------- #


change_dynamics_velocity_dict = {j_id: {'maxJointVelocity': 0.00000001} for j_id in range(7)}
low_max_force, high_max_force = 0.00000001, 0.00000001
j_ids_and_forces = [
    (0, high_max_force), (1, high_max_force), (2, high_max_force),
    (3, low_max_force), (4, low_max_force), (5, low_max_force)
]
change_dynamics_max_force_dict = {j_id: {'jointLimitForce': max_f} for j_id, max_f in j_ids_and_forces}

joint_damping = 0.000000001
fixed_linear_damping, fixed_angular_damping = 0., 0.
change_dynamics_j_damping_dict = {j_id: {'jointDamping': joint_damping,
                                         'linearDamping': fixed_linear_damping,
                                         'angularDamping': fixed_angular_damping}
                                  for j_id in range(7)}

# Joint limits
joint_low_up_lim_j_id_arm = {
    0: (-1.0471975, 3.14),
    4: (0., 3.14)
}

change_dynamics_j_lims_dict = {
    j_id: {'jointLowerLimit': joint_low_up_lim_j_id_arm[j_id][0],
           'jointUpperLimit': joint_low_up_lim_j_id_arm[j_id][1]}
    for j_id in joint_low_up_lim_j_id_arm
}
CHANGE_DYNAMICS_DICT = {**change_dynamics_velocity_dict, **change_dynamics_max_force_dict,
                        **change_dynamics_j_damping_dict, **change_dynamics_j_lims_dict}



# ---------------------------------------------- #
#                    SCENE
# ---------------------------------------------- #


DEFAULT_OBJECT_POSE_XYZ = [0.80, 0.95, -0.18318535463122008]
DEFAULT_OBJECT_ORIENT_XYZW = [ 0, 0, 0.7512804, 0.6599831 ] #[0, 0, 0, 1] # [0, 0, -0.3826834, 0.9238795]

UR5_SIH_SCHUNK_CENTER_WORKSPACE = 0
UR5_SIH_SCHUNK_WORKSPACE_RADIUS = 1



