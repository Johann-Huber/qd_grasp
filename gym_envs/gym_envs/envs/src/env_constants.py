

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




# ====================================================================================== #
#                                   RELATIVE PATHS
# ====================================================================================== #

GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS = 'envs/3d_models/robots'



