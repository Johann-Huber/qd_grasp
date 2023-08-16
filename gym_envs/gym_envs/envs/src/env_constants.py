

from enum import Enum


# ====================================================================================== #
#                                   SUPPORTED ROBOTS
# ====================================================================================== #


SimulatedRobot = Enum(
    'SimulatedRobot',
    ['KUKA_WSG50',
     'KUKA_ALLEGRO']
)

INPUT_ARG_ROBOT2ROBOT_TYPE_NAME = {
    'kuka_wsg50': SimulatedRobot.KUKA_WSG50,
    'kuka_allegro': SimulatedRobot.KUKA_ALLEGRO,
}

ROBOT_KWARGS = {
    SimulatedRobot.KUKA_WSG50: {
        'gym_env_name': 'kuka_wsg50_grasping-v0',
        'gene_per_keypoints': 6,
        'link_id_contact': None,
        'nb_steps_to_rollout': 10,
        'nb_iter_ref': 2000,
        'n_it_closing_grip': 25,
        'controller_type': 'interpolate keypoints cartesian speed control grip',
    },

    SimulatedRobot.KUKA_ALLEGRO: {
        'gym_env_name': 'kuka_allegro_grasping-v0',
        'gene_per_keypoints': 6,  # cartesian : 6-DOF [x, y, z, theta_1, theta_2, theta_3]
        'link_id_contact': None,
        'nb_steps_to_rollout': 10,
        'nb_iter_ref': 1500,
        'n_it_closing_grip': 50,
        'controller_type': 'interpolate keypoints cartesian finger synergies',
    },
}

ROBOT_TYPE_SKIP_REDEPLOIEMENT_SAFECHECK = [SimulatedRobot.KUKA_WSG50, SimulatedRobot.KUKA_ALLEGRO]


# ====================================================================================== #
#                             GRASPING ENVIRONMENTS PARAMETERS
# ====================================================================================== #


# ---------------------------------------------- #
#                    SETUP
# ---------------------------------------------- #

TABLE_HEIGHT = 0.76



# ---------------------------------------------- #
#                    KUKA
# ---------------------------------------------- #

KUKA_WS_RADIUS = 1.2


KUKA_DEFAULT_INIT_OBJECT_POSITION = [0., 0.15, -0.18318535463122008]

KUKA_JOINT_IDS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 13]

KUKA_CONTACT_IDS = [8, 9, 10, 11, 12, 13]


KUKA_N_CONTROL_GRIPPER = 4
KUKA_END_EFFECTOR_ID = 6 #7 #14
KUKA_CENTER_WORKSPACE = 0
KUKA_DISABLED_COLLISION_PAIRS = [[8, 11], [10, 13]]


KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS = [
    -0.3339413562255018, 0.379227146702238, -0.8712221477875728, 1.2162870375730164, -0.20887871697148447,
    -0.9065636946584263, 0.00015004095036508916
]
KUKA_ABOVE_OBJECT_INIT_POSITION_GRIPPER = [
    -0.04999999999999999, -0.00105972150624707, -0.014672116051631068, 0.016698095337863785
]
KUKA_ABOVE_OBJECT_INIT_POSITION = KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS + KUKA_ABOVE_OBJECT_INIT_POSITION_GRIPPER



KUKA_INIT_POS_X_MIN, KUKA_INIT_POS_X_MAX = -0.5, 0.5
KUKA_INIT_POS_Y_MIN, KUKA_INIT_POS_Y_MAX = 0, 0.5
KUKA_INIT_POS_Z_MIN, KUKA_INIT_POS_Z_MAX = -0.3, 0.2


# ---------------------------------------------- #
#                 KUKA ALLEGRO
# ---------------------------------------------- #

KUKA_ALLEGRO_JOINT_IDS = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27]

KUKA_ALLEGRO_INIT_POS_X_MIN, KUKA_ALLEGRO_INIT_POS_X_MAX = -0.3, 0.3
KUKA_ALLEGRO_INIT_POS_Y_MIN, KUKA_ALLEGRO_INIT_POS_Y_MAX = -0.1, 0.3
KUKA_ALLEGRO_INIT_POS_Z_MIN, KUKA_ALLEGRO_INIT_POS_Z_MAX = -1.0, -0.5



# ---------------------------------------------- #
#         FITNESS NORMALIZATION BOUNDARIES
# ---------------------------------------------- #

# max/min energy fit considered for adressed experimnent. Deduced from success repertoires generated with DC_NSMBS
ENERGY_FIT_NORM_BOUNDS_PER_ROB = {
    SimulatedRobot.KUKA_WSG50: {
        'max': -10000.,
        'min': -30000.,
    },
    SimulatedRobot.KUKA_ALLEGRO: {
        'max': -5000.,
        'min': -25000.,
    }
}









