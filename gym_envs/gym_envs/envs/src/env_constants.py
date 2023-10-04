

from enum import Enum


# ====================================================================================== #
#                                   SUPPORTED ROBOTS
# ====================================================================================== #


SimulatedRobot = Enum(
    'SimulatedRobot',
    [
        'KUKA_WSG50',
        'KUKA_ALLEGRO',
        'PANDA_2_FINGERS',
        'UR5_SIH_SCHUNK',
        'BX_2_FINGERS'
     ]
)

INPUT_ARG_ROBOT2ROBOT_TYPE_NAME = {
    'kuka_wsg50': SimulatedRobot.KUKA_WSG50,
    'kuka_allegro': SimulatedRobot.KUKA_ALLEGRO,
    'panda_2f': SimulatedRobot.PANDA_2_FINGERS,
    'ur5_sih_schunk': SimulatedRobot.UR5_SIH_SCHUNK,
    'baxter_2f': SimulatedRobot.BX_2_FINGERS
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

    SimulatedRobot.PANDA_2_FINGERS: {
        'gym_env_name': 'panda_2f_grasping-v0',
        'gene_per_keypoints': 6,
        'link_id_contact': None,
        'nb_steps_to_rollout': 10,
        'nb_iter_ref': 900,
        'n_it_closing_grip': 25,
        'controller_type': 'interpolate keypoints cartesian speed control grip',
    },

    SimulatedRobot.UR5_SIH_SCHUNK: {
        'gym_env_name': 'ur5_sih_schunk_grasping-v0',
        'gene_per_keypoints': 6,
        'link_id_contact': None,
        'nb_steps_to_rollout': 4,
        'nb_iter_ref': 700,
        'n_it_closing_grip': 40,
        'controller_type': 'interpolate keypoints cartesian finger synergies',
    },

    SimulatedRobot.BX_2_FINGERS: {
        'gym_env_name': 'bx_2f_grasping-v0',
        'gene_per_keypoints': 6,
        'link_id_contact': None,
        'nb_steps_to_rollout': 10,
        'nb_iter_ref': 2000,
        'n_it_closing_grip': 40,
        'controller_type': 'interpolate keypoints cartesian speed control grip',
    },

}
ROBOT_TYPE_SKIP_REDEPLOIEMENT_SAFECHECK = [
    SimulatedRobot.KUKA_WSG50,
    SimulatedRobot.KUKA_ALLEGRO,
    SimulatedRobot.PANDA_2_FINGERS,
    SimulatedRobot.UR5_SIH_SCHUNK,
    SimulatedRobot.BX_2_FINGERS,
]


# ====================================================================================== #
#                             GRASPING ENVIRONMENTS PARAMETERS
# ====================================================================================== #


# ---------------------------------------------- #
#                    SETUP
# ---------------------------------------------- #

TABLE_HEIGHT = 0.76


TableLabel = Enum(
    'TableLabel',
    [
        'STANDARD_TABLE',
        'UR5_TABLE',
     ]
)

# Matching table

sim_robot2table = {
    SimulatedRobot.KUKA_WSG50: TableLabel.STANDARD_TABLE,
    SimulatedRobot.KUKA_ALLEGRO: TableLabel.STANDARD_TABLE,
    SimulatedRobot.PANDA_2_FINGERS: TableLabel.STANDARD_TABLE,
    SimulatedRobot.UR5_SIH_SCHUNK: TableLabel.UR5_TABLE,
    SimulatedRobot.BX_2_FINGERS: TableLabel.STANDARD_TABLE,
}





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
    },
    SimulatedRobot.PANDA_2_FINGERS: {
        'max': 0.,
        'min': -40000.
    },
    SimulatedRobot.UR5_SIH_SCHUNK: {
        'max': 0.,
        'min': -40000.
    },
    SimulatedRobot.BX_2_FINGERS: {
        'max': 0.,
        'min': -40000.
    },
}




# ====================================================================================== #
#                                   RELATIVE PATHS
# ====================================================================================== #

GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS = 'envs/3d_models/robots'



# ====================================================================================== #
#                                   ENV SPECIFIC VARS
# ====================================================================================== #

# Baxter

BAXTER_BASED_SUPPORTED_ENVS = [
    SimulatedRobot.BX_2_FINGERS
]


# ====================================================================================== #
#                                   OPERATIONAL SPACE
# ====================================================================================== #

# boundaries for genome-to-6DoFpose conversion (cartesian controllers)

CARTESIAN_SCENE_POSE_BOUNDARIES = {
    'kuka_allegro_grasping-v0': {
        'min_x_val': -0.4, 'max_x_val': 0.4,
        'min_y_val': -0.1, 'max_y_val': 0.3,
        'min_z_val': -0.3, 'max_z_val': 0.1,
    },
    'kuka_wsg50_grasping-v0': {
        'min_x_val': -0.5, 'max_x_val': 0.5,
        'min_y_val': -0.2, 'max_y_val': 0.5,
        'min_z_val': -0.2, 'max_z_val': 0.3,
    },

    'panda_2f_grasping-v0': {
        'min_x_val': -0.25, 'max_x_val': 0.25,
        'min_y_val': 0.15, 'max_y_val': 0.40,
        'min_z_val': -0.23, 'max_z_val': 0.037,
    },

    'ur5_sih_schunk_grasping-v0': {
        'min_x_val': 0.65, 'max_x_val': 0.95,
        'min_y_val': 0.7, 'max_y_val': 1.05,
        'min_z_val': -0.18, 'max_z_val': 0.02,
    },
    'bx_2f_grasping-v0': {
        'min_x_val': -0.4, 'max_x_val': 0.4,
        'min_y_val': -0.1, 'max_y_val': 0.3,
        'min_z_val': -0.3, 'max_z_val': 0.1,
    }

}

MIN_X_VAL, MAX_X_VAL = -0.5, 0.5
MIN_Y_VAL, MAX_Y_VAL = -0.2, 0.5
MIN_Z_VAL, MAX_Z_VAL = -0.2, 0.3

MIN_X_TOUCH_VAL, MAX_X_TOUCH_VAL = MIN_X_VAL, MAX_X_VAL
MIN_Y_TOUCH_VAL, MAX_Y_TOUCH_VAL = MIN_Y_VAL, MAX_Y_VAL
MIN_Z_TOUCH_VAL, MAX_Z_TOUCH_VAL = MIN_Z_VAL, MAX_Z_VAL

MIN_X_VERIF_VAL, MAX_X_VERIF_VAL = MIN_X_VAL, MAX_X_VAL
MIN_Y_VERIF_VAL, MAX_Y_VERIF_VAL = MIN_Y_VAL, MAX_Y_VAL
MIN_Z_VERIF_VAL, MAX_Z_VERIF_VAL = MIN_Z_VAL, MAX_Z_VAL

MIN_EULER_VAL, MAX_EULER_VAL = -2*3.14, 2*3.14





