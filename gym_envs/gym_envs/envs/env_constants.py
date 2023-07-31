


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
    'kuka_ik': {
        'max': -10000.,
        'min': -30000.,
    },
    'kuka_iiwa_allegro_ik': {
        'max': -5000.,
        'min': -25000.,
    }
}








