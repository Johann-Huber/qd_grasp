
import numpy as np


# ---------------------------------------------- #
#                  JOINT IDS
# ---------------------------------------------- #

# Definition

BX_LEFT_ARM_JOINT_ID_STATUS = {
    0: {'name': 'left_s0', 'status': 'CONTROLLED', 'is_controllable': True},
    1: {'name': 'left_s1', 'status': 'CONTROLLED', 'is_controllable': True},
    2: {'name': 'left_e0', 'status': 'CONTROLLED', 'is_controllable': True},
    3: {'name': 'left_e1', 'status': 'CONTROLLED', 'is_controllable': True},
    4: {'name': 'left_w0', 'status': 'CONTROLLED', 'is_controllable': True},
    5: {'name': 'left_w1', 'status': 'CONTROLLED', 'is_controllable': True},
    6: {'name': 'left_w2', 'status': 'CONTROLLED', 'is_controllable': True},
}

BX_LEFT_GRIP_JOINT_ID_STATUS = {
    7: {'name': 'l_gripper_r_finger_joint', 'status': 'CONTROLLED', 'part': 'gripper', 'is_controllable': True},
    8: {'name': 'l_gripper_l_finger_joint', 'status': 'CONTROLLED', 'part': 'gripper', 'is_controllable': True},
}




# Key variables

# --- General

BX_LEFT_ARM_ALL_JOINT_IDS = [j_id for j_id in BX_LEFT_ARM_JOINT_ID_STATUS]
BX_2_FINGERS_GRIP_ALL_JOINT_IDS = [j_id for j_id in BX_LEFT_GRIP_JOINT_ID_STATUS]

# --- Controlled

BX_LEFT_ARM_CONTROLLED_JOINT_IDS = [
    j_id for j_id in BX_LEFT_ARM_JOINT_ID_STATUS
    if BX_LEFT_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
]

BX_2_FINGERS_GRIP_CONTROLLED_JOINT_IDS = [
    j_id for j_id in BX_LEFT_GRIP_JOINT_ID_STATUS
    if BX_LEFT_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
]

# --- Controllable

BX_LEFT_ARM_CONTROLLABLE_JOINT_IDS = [
    j_id for j_id in BX_LEFT_ARM_JOINT_ID_STATUS
    if BX_LEFT_ARM_JOINT_ID_STATUS[j_id]['is_controllable']
]

BX_2_FINGERS_GRIP_CONTROLLABLE_JOINT_IDS = [
    j_id for j_id in BX_LEFT_GRIP_JOINT_ID_STATUS
    if BX_LEFT_GRIP_JOINT_ID_STATUS[j_id]['is_controllable']
]




# ---------------------------------------------- #
#                    SCENE
# ---------------------------------------------- #


BX_DEFAULT_INIT_OBJECT_POSITION = [-0.000309649771714305, 0.12144435090106637, -0.22318535463122008]


# ---------------------------------------------- #
#                    PATHS
# ---------------------------------------------- #

BX_2_FINGERS_RELATIVE_PATH_XACRO = 'baxter_description/urdf/baxter_symmetric.xacro'
BX_2_FINGERS_RELATIVE_PATH_GENERATED_URDF = 'generated/'


#y_offseted_obj_names
#y_offseted_obj_names

POSE_OFFSETED_OBJ_NAMES = [
    'ycb_apple',
    'ycb_extra_large_clamp',
    'ycb_rubiks_cube',
    'ycb_strawberry',
    'ycb_adjustable_wrench',
    'ycb_chips_can',
    'ycb_orange',
    'ycb_baseball',
    'ycb_tennis_ball',
    'ycb_tomato_soup_can',
    'ycb_plum',
]
OFFSET_OBJ_POSE = np.array([-0.1, 0.1, 0])

# ---------------------------------------------- #
#                 INITIAL STATE
# ---------------------------------------------- #


BX_BASE_POSITION = [0, -0.5, -.074830]
BX_BASE_ORIENTATION = [0, 0, -1, -1]

BX_FIXED_JOINTS_RIGHT_ARM_DICT = {
    'right_s0': 0.08, 'right_s1': -1.0, 'right_e0': 1.19, 'right_e1': 1.94, 'right_w0': -0.67,
    'right_w1': 1.03, 'right_w2': -0.50, 'r_gripper_l_finger_joint': 0, 'r_gripper_r_finger_joint': 0
}
BX_UNTUCK_JOINT_POSITIONS_RIGHT_DICT = {12: 0.08, 13: -1.0, 14: 1.19, 15: 1.94, 16: -0.67, 18: 1.03, 19: 0.50}

BX_FIXED_JOINTS_LEFT_ARM_DICT = {
    'left_s0': -0.08, 'left_s1': -1.0, 'left_e0': -1.19, 'left_e1': 1.94, 'left_w0': 0.67,
    'left_w1': 1.03, 'left_w2': -0.50, 'l_gripper_l_finger_joint': 0, 'l_gripper_r_finger_joint': 0
}
BX_UNTUCK_JOINT_POSITIONS_LEFT_DICT = {34: -0.08, 35: -1.0, 36: -1.19, 37: 1.94, 38: 0.67, 40: 1.03, 41: -0.50}



BX_DISABLED_COLLISION_PAIRS_FIXED_ARM = [[-1, 1], [-1, 2], [0, 2], [1, 3], [1, 4], [7, 8], [3, 5], [3, 6]]

BX_DISABLED_COLLISION_PAIRS_DEFAULT = [
    [49, 51], [38, 53], [27, 29], [16, 31], [1, 10], [1, 7], [1, 5], [0, 10], [0, 7], [0, 5], [0, 1],
    [40, 53], [37, 54], [34, 36], [18, 31], [15, 32], [12, 14], [35, 2], [34, 2], [14, 2], [13, 2], [12, 2],
    [2, 7], [1, 2], [0, 2], [41, 53], [36, 2], [34, 54], [54, 2], [50, 55], [38, 54], [1, 53], [1, 38],
    [1, 37], [16, 32], [19, 31], [49, 52], [50, 51], [50, 52]
]

BX_FIXED_ARM_CHANGE_LATERAL_FRICTION_IDS = [7, 8]
BX_DEFAULT_CHANGE_LATERAL_FRICTION_IDS = [28, 30, 50, 52]

BX_END_EFFECTOR_ID_FIXED_ARM = 8
BX_END_EFFECTOR_ID_DEFAULT_LEFT = 48
BX_END_EFFECTOR_ID_DEFAULT_RIGHT = 26


BX_JOINT_IDS_FIXED_ARM = [0, 1, 2, 3, 4, 5, 6, 8, 7]
BX_JOINT_IDS_DEFAULT_LEFT = [34, 35, 36, 37, 38, 40, 41, 49, 51]
BX_JOINT_IDS_DEFAULT_RIGHT = [12, 13, 14, 15, 16, 17, 18, 19, 27, 29]

BX_WS_RADIUS = 1.2

BX_CENTER_WORKSPACE_FIXED_ARM = 1
BX_CENTER_WORKSPACE_DEFAULT_LEFT = 34
BX_CENTER_WORKSPACE_DEFAULT_RIGHT = 13

BX_CONTACT_IDS_FIXED_ARM = [7, 8]
BX_CONTACT_IDS_DEFAULT_LEFT = [47, 48, 49, 50, 51, 52]
BX_CONTACT_IDS_DEFAULT_RIGHT = [27, 28, 29, 20]


BX_INIT_POS_X_MIN, BX_INIT_POS_X_MAX = -1.0, 0.2
BX_INIT_POS_Y_MIN, BX_INIT_POS_Y_MAX = -0.1, 0.2
BX_INIT_POS_Z_MIN, BX_INIT_POS_Z_MAX = -0., 0.2

REAL_BX_INIT_POS_Z_MIN, REAL_BX_INIT_POS_Z_MAX = -0.2, 0.0


BX_GRIP_SLOT_PER_OBJECTS = {
    'ycb_hammer': 2,
    'ycb_power_drill': 2,
    'ycb_pudding_box': 4,
    'ycb_knife': 1,
    'ycb_small_clamp': 1,
    'ycb_large_clamp': 1,
    'ycb_rubiks_cube': 4,
    'ycb_large_marker': 1,
    'ycb_scissors': 1,
    'ycb_lemon': 2,
    'ycb_skillet_lid': None,
    'ycb_potted_meat_can': 4,
    'ycb_racquetball': 3,
    'ycb_foam_brick': 3,
    'ycb_bleach_cleanser': 3,
    'ycb_skillet': None,
    'ycb_adjustable_wrench': 1,
    'ycb_master_chef_can': None,
    'ycb_apple': 3,
    'ycb_medium_clamp': 1,
    'ycb_banana': 2,
    'ycb_mini_soccer_ball': None,
    'ycb_softball': None,
    'ycb_baseball': 3,
    'ycb_mug': 1,
    'ycb_spatula': 1,
    'ycb_bowl': 1,
    'ycb_spoon': 1,
    'ycb_padlock': 3,
    'ycb_strawberry': 3,
    'ycb_chips_can': 4,
    'ycb_sugar_box': 3,
    'ycb_cracker_box': 4,
    'ycb_pear': 3,
    'ycb_philips_screwdriver': 2,
    'ycb_timer': 4,
    'ycb_extra_large_clamp': 1,
    'ycb_pitcher_base': None,
    'ycb_tomato_soup_can': 3,
    'ycb_flat_screwdriver': 3,
    'ycb_tuna_fish_can': 4,
    'ycb_windex_bottle': 2,
    'ycb_fork': 1,
    'ycb_gelatin_box': 2,
    'ycb_plum': 2,
    'ycb_wood_block': None,
    'ycb_orange': 3,
    'ycb_golf_ball': 3,
    'ycb_sponge': 4,
    'ycb_chain': 1,
    'ycb_peach': 3,
    'ycb_tennis_ball': 3,
    'ycb_plate': 1,
}

BX_ABOVE_OBJECT_INIT_POSITION_JOINTS = [
    -0.3619402539524409,
    -0.5276636100805383,
    -0.9546725367813972,
    1.376256330053032,
    0.8689134849558748,
    0.5371586494496319,
    -0.48412196579064015
]

BX_DEFAULT_LIMIT_SCALE = 0.13
BX_DEFAULT_FINGER = "extended_narrow"
BX_DEFAULT_TIP = "basic_soft"
BX_DEFAULT_GRASP = "inner"



