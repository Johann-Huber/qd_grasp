
import pdb

import pathlib
from enum import Enum
import os



# Variant names
POP_BASED_BASELINES_ALGO_VARIANTS = ['random', 'fit', 'ns']
NSMBS_ALGO_VARIANTS = ['nsmbs']
CMA_MAE_ALGO_VARIANTS = ['cma_mae']
CMA_ME_ALGO_VARIANTS = ['cma_me']
CMA_ES_ALGO_VARIANTS = ['cma_es']


QD_VARIANTS = ['nslc', 'me_scs', 'me_fit', 'me_rand', 'me_nov', 'me_nov_scs', 'me_nov_fit']

PYRIBS_QD_VARIANTS = ['cma_mae', 'cma_es', 'cma_me']
SERENE_QD_VARIANTS = ['serene']


ELITE_STRUCTURED_ARCHIVE_ALGO_VARIANTS = ['me_scs', 'me_rand', 'me_fit']
ELITE_NOVELTY_STRUCTURED_ARCHIVE_ALGO_VARIANTS = ['me_nov', 'me_nov_scs', 'me_nov_fit']
NOVELTY_ARCHIVE_ALGO_VARIANTS = ['ns', 'nsmbs']
ARCHIVE_LESS_ALGO_VARIANTS = ['random', 'fit']

POP_BASED_RANDOM_SELECTION_ALGO_VARIANTS = POP_BASED_BASELINES_ALGO_VARIANTS + NSMBS_ALGO_VARIANTS + SERENE_QD_VARIANTS
#pdb.set_trace()

ELITE_STRUCTURED_ARCHIVE_SUCCESS_BASED_SELECTION_ALGO_VARIANTS = ['me_scs']
ARCHIVE_BASED_RANDOM_SELECTION_ALGO_VARIANTS = ['me_rand']
ARCHIVE_BASED_FITNESS_SELECTION_ALGO_VARIANTS = ['me_fit']
ARCHIVE_BASED_NOVELTY_SELECTION_ALGO_VARIANTS = ['me_nov']
ARCHIVE_BASED_NOVELTY_SUCCESS_SELECTION_ALGO_VARIANTS = ['me_nov_scs']
ARCHIVE_BASED_NOVELTY_FITNESS_SELECTION_ALGO_VARIANTS = ['me_nov_fit']
NSLC_NSGA_II_SELECTION_ALGO_VARIANTS = ['nslc']

SelectOffspringStrategy = Enum(
    'SelectOffspringStrategy',
    ['RANDOM_FROM_POP',
     'RANDOM_FROM_ARCHIVE',
     'NOVELTY_FROM_ARCHIVE',
     'FITNESS_FROM_ARCHIVE',
     'NOVELTY_SUCCESS_FROM_ARCHIVE',
     'NOVELTY_FITNESS_FROM_ARCHIVE',
     'SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE',
     'FORCE_SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE',
     'NSLC']
)

SUPPORTED_VARIANTS_NAMES = POP_BASED_BASELINES_ALGO_VARIANTS + NSMBS_ALGO_VARIANTS + QD_VARIANTS + PYRIBS_QD_VARIANTS + SERENE_QD_VARIANTS

# Conditions on variants
VARIANTS_NO_EVO_PROCESS_REINIT = []

NSLC_ALGO_VARIANTS = ['nslc']

MAP_ELITES_ALGO_VARIANTS = ['me_scs', 'me_rand', 'me_nov', 'me_nov_scs', 'me_nov_fit', 'me_fit']

POP_BASED_ALGO_VARIANTS = POP_BASED_BASELINES_ALGO_VARIANTS + NSMBS_ALGO_VARIANTS + NSLC_ALGO_VARIANTS + SERENE_QD_VARIANTS

WITH_NOVELTY_POP_BASED_ALGO_VARIANTS = ['ns', 'nsmbs', 'serene']


#----------------------------------------------------------------------------------------------------------------------#
# VARIANTS HYPERPARAMETERS
#----------------------------------------------------------------------------------------------------------------------#

# Evolutionary proccess associated to each variant
ALGO_EVO_PROCESS = {'nsmbs': 'ns_rand_multi_bd',
                    'random': 'random_search',
                    'ns': 'ns_nov',
                    'fit': 'fitness',
                    'cma_mae': 'cma_mae',
                    'cma_es': 'cma_es',
                    'cma_me': 'cma_me',
                    'serene': 'serene',
                    'nslc': 'ns_local_competition',
                    'me_scs': 'map_elites',
                    'me_fit': 'map_elites',
                    'me_rand': 'map_elites',
                    'me_nov': 'map_elites',
                    'me_nov_scs': 'map_elites',
                    'me_nov_fit': 'map_elites',
                    }

EVO_PROCESS_WITH_LOCAL_COMPETITION = ['ns_local_competition']
MULTI_BD_EVO_PROCESSES = {'ns_rand_multi_bd'}
NO_NOVELTY_EVO_PROCESSES = {'random_search', 'fitness', 'map_elites', 'cma_mae', 'cma_es', 'cma_me'}
CMA_BASED_EVO_PROCESSES = {'cma_mae', 'cma_es', 'cma_me'}
SERENE_BASED_EVO_PROCESSES = {'serene'}

# Mutation flag associated to each variant
ALGO_MUT_FLGS = {'nsmbs': 'gauss',
                 'random': 'gauss',
                 'fit': 'gauss',
                 'ns': 'gauss',
                 'nslc': 'gauss',
                 'me_scs': 'gauss',
                 'cma_mae': 'pyribs_mut',
                 'cma_es': 'pyribs_mut',
                 'cma_me': 'pyribs_mut',
                 'serene': 'serene',
                 'me_fit': 'gauss',
                 'me_rand': 'gauss',
                 'me_nov': 'gauss',
                 'me_nov_scs': 'gauss',
                 'me_nov_fit': 'gauss'}

SUPPORTED_MUT_FLGS = {'gauss', 'pyribs_mut', 'serene'}

# Novely metadata associated to each bd_flg

BD_FLG = 'pos_touch'

BD_BOUNDS_END_EFFECTOR_POS = [-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5]

BD_METAPARAMS = {
    'pos_touch': {
        'bd_bounds': [[-0.35, 0.35], [-0.15, 0.2], [-0.2, 0.5]],
        'bd_indexes': [0, 0, 0],
        'novelty_metric': ['minkowski'] * 1
    },
}

BD_FLG_TO_BD_NAMES = {
    'pos_touch': [
        'pos_touch_time'
    ],
}

FIXED_VALUE_UNDEFINED_BD_FILL = 0.

# Clustering and distance computation
DIFF_OR_THRESH = 0.4  # threshold for clustering grasping orientations
K_NN_NOV = 15  # number of nearest neighbours for novelty computation
INF_NN_DIST = 1000000000  # for security against infinite distances in KDtree queries

# Initialisation process
REFILL_POP_METHOD = 'copies'


#----------------------------------------------------------------------------------------------------------------------#
# GENOMES
#----------------------------------------------------------------------------------------------------------------------#
BOUND_GENOTYPE_THRESH = 1.


#----------------------------------------------------------------------------------------------------------------------#
# EVALUATION
#----------------------------------------------------------------------------------------------------------------------#

AUTO_COLLIDE = True
N_DIGITS_AROUND_INDS_EVAL = 3

N_ITER_STABILIZING_SIM = 0
N_ITER_STABILIZING_SIM_DEFAULT_ROBOT_GRASP = 700 #2000 #10000


#----------------------------------------------------------------------------------------------------------------------#
# CONTROLLER
#----------------------------------------------------------------------------------------------------------------------#

SUPPORTED_CONTROLLERS = {
    'interpolate keypoints speed control grip',
    'interpolate keypoints finger synergies'
}
NB_KEYPOINTS = 3


#----------------------------------------------------------------------------------------------------------------------#
# EVOLUTIONARY PROCESS
#----------------------------------------------------------------------------------------------------------------------#

# NS process intialization trials
N_TRIALS_SAFEGUARD = False
N_TRY_NS_MAX = 10

# Population reinitialization
N_GEN_NO_SUCCESS_REINIT_RESEARCH_FREQ = 500

# Population regeneration
N_GEN_REGEN_FREQ = 3  # 100
NOV_REGEN_WEIGHT = 1
K_NN_NOV_REGEN = 15  # number of nearest neighbours for novelty computation (regeneration of successful inds)

# Mutation process
SIGMA = 0.5 #0.02  # std of the mutation of one gene
SIGMA_PREHENSION_GRIP_TIME_PERTURBATION = [0.2]
SIGMA_GRASP_PRIMITIVE_LABEL_PERTURBATION = [0.5] # should jump from one label to another in a single mutation

CXPB = 0. # probability with which two individuals are crossed (ONLY FOR QD PART)
MUTPB = 1.0  # probability for mutating an individual

# Selection process
OFFSPRING_NB_COEFF = 1. # number of offsprings generated (coeff of pop length)
PB_ADD2ARCHIVE = 0.4 # if ARCHIVE is random, probability to add individual to archive
TOURNSIZE = 15  # drives towards selective pressure
TOURNSIZE_RATIO = 0.15
RANDOM_SEL_FLG = True  # decide if selection is random or based on tournament

POP_RATIO_QD_PARETO_SEL_TOURNSIZE = 1.  # if pop_size=100 and n_added_inds_archive=12, size(archive) > 100 @n_gen=10


# Archive size management
SUPPORTED_ARCHIVE_LIMIT_STRAT = ['random']
ARCHIVE_LIMIT_STRAT = 'random'
ARCHIVE_LIMIT_SIZE = 25000  #5000 #10000
ARCHIVE_DECREMENTAL_RATIO = 0.9  # if archive size is bigger than thresh, cut down archive by this ratio
GMM_N_COMP = 4  # number of GMM components in case of gmm sampling archive management

EVO_PROCESS_ARCHIVE_FILL_NOV = ['ns_nov', 'ns_rand_multi_bd', 'ns_local_competition']
EVO_PROCESS_ARCHIVE_LESS = ['random_search', 'fitness']
EVO_PROCESS_ARCHIVE_STRUCTURED_ELITES = ['map_elites']

# Saving condition
SUCCESS_CRITERION = 'is_success'
SAVE_IND_COND = SUCCESS_CRITERION


#----------------------------------------------------------------------------------------------------------------------#
# MONITORING
#----------------------------------------------------------------------------------------------------------------------#

DO_MEASURES = True

# Timer
NS_RUN_TIME_LABEL = 'run_ns'
QD_RUN_TIME_LABEL = 'run_qd'

# SIMULATION
RESET_MODE = True  # True : do not load from scratch bullet env but restore state
INVALIDATE_CONTACT_TABLE = False

GRASP_WHILE_CLOSE_TOLERANCE = 2


#----------------------------------------------------------------------------------------------------------------------#
# RUNNING DATA SAVING
#----------------------------------------------------------------------------------------------------------------------#

# Frequent dump
DUMP_SCS_ARCHIVE_ON_THE_FLY = True
N_GEN_FREQ_DUMP_SCS_ARCHIVE = 10

DUMP_QD_ARCHIVE_ON_THE_FLY = True
N_GEN_FREQ_DUMP_QD_ARCHIVE = 10

# Data storage
N_SCS_RUN_DATA_KEY = 'Number of successful individuals (archive_success len)'

#----------------------------------------------------------------------------------------------------------------------#
# POST PROCESSING
#----------------------------------------------------------------------------------------------------------------------#

# Plotting
EXPORT_PLOT = True
SUPPORTED_DUMP_PATH_TYPES = {str, pathlib.PosixPath}

SUPPORTED_TIMER_FORMATS = {'seconds', 'h:m:s'}


# Execution
RETURN_SUCCESS_CODE = 0
RETURN_FAILURE_CODE = 1


#----------------------------------------------------------------------------------------------------------------------#
# COMPUTATIONAL UTILITIES
#----------------------------------------------------------------------------------------------------------------------#

INF_FLOAT_CONST = float('inf')



#----------------------------------------------------------------------------------------------------------------------#
# BULLET ENV HYPERPARAMETERS
#----------------------------------------------------------------------------------------------------------------------#

CAMERA_DEFAULT_PARAMETERS = {
    'target': [-0.2, 0, 0.3],
    'distance': 0.8,
    'yaw': 170,
    'pitch': -20
}

GRAVITY_FLG = True

ACTION_UPPER_BOUND = 1

DEFAULT_INVALID_ID_BULLET = -1

GRIPPER_DISPLAY_LINE_WIDTH = 4

BULLET_PLANE_URDF_FILE_RPATH = "plane.urdf"
BULLET_TABLE_URDF_FILE_RPATH = "table/table.urdf"

#----------------------------------------------------------------------------------------------------------------------#
# BULLET REAL SCENE
#----------------------------------------------------------------------------------------------------------------------#

REAL_SCENE = True  #False
LOCAL_REAL_SCENE = os.getcwd() + '/pybullet_data/'
ROOT_REAL_SCENE = LOCAL_REAL_SCENE
BULLET_TABLE_URDF_FILE_RPATH_REAL_SCENE = ROOT_REAL_SCENE + 'table/table.urdf'

LOCAL_PATH_SIM2REAL_SCENE_FLG = REAL_SCENE  # trigger sim2real scene

REAL_SCENE_TABLE_HEIGHT = 0.7435

REAL_SCENE_TABLE_TOP = 0.7838

BULLET_TABLE_BASE_ORIENTATION = [0, 0, 0, 1]

BULLET_OBJECT_DEFAULT_POSITION = [-0.000309649771714305, 0.12144435090106637, -0.22318535463122008]

BULLET_OBJECT_DEFAULT_ORIENTATION = [0, 0, 0, 1]  # initial orientation of the object during loading


BULLET_DEFAULT_N_STEPS_TO_ROLL = 1  # nb time to call p.stepSimulation within one step (depends on the robot)

BULLET_DEFAULT_DISPLAY_FLG = False  # whether to display steps
BULLET_DEFAULT_GRIP_DISPLAY_FLG = False  # whether to display the end effector trajectory

ENV_DEFAULT_INIT_STATE = None

BULLET_DUMMY_OBS = [None]  # returned observation vector when object is not initialized


#----------------------------------------------------------------------------------------------------------------------#
# QUALITY
#----------------------------------------------------------------------------------------------------------------------#


# NORMALIZATION
# (bellow values have been defined w.r.t. measured limits of distributions on preliminary experimentS)
FITNESS_OBJECT_STATE_VARIANCE_MIN = -0.1
FITNESS_OBJECT_STATE_VARIANCE_MAX = 0.
FITNESS_TOUCH_VARIANCE_MIN = -200
FITNESS_TOUCH_VARIANCE_MAX = 0


APPROACH_THRESH_DIST = 0.2
PREHENSION_THRESH_DIST = 0.05


SUPPORTED_PREHENSION_CRITERIA_STR = ['mono_eval']

# NSLC
K_NN_LOCAL_QUALITY = 50

#----------------------------------------------------------------------------------------------------------------------#
# BOUNDARIES FOR GENOME-TO-6DoFpose conversion (cartesian controllers)
#----------------------------------------------------------------------------------------------------------------------#


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


#----------------------------------------------------------------------------------------------------------------------#
# CMA-MAE
#----------------------------------------------------------------------------------------------------------------------#

CMA_MAE_EMITTER_BATCH_SIZE = 36
CMA_MAE_N_EMITTERS = 15
CMA_MAE_POP_SIZE = CMA_MAE_EMITTER_BATCH_SIZE * CMA_MAE_N_EMITTERS

CMA_MAE_PREDEFINED_ALPHA = 0.01
CMA_ME_PREDEFINED_ALPHA = 1.0
CMA_ES_PREDEFINED_ALPHA = 0.0


#----------------------------------------------------------------------------------------------------------------------#
# SERENE
#----------------------------------------------------------------------------------------------------------------------#

serene_chunk_size = 1000
emitter_population_len = 6
serene_emitter_local_improvement_flg = False
serene_k_novelty_neighs = 15
serene_novelty_distance_metric = 'euclidean'
serene_agent_template = {
    'genome': None,
    'reward': None,
    'bd': None,
    'novelty': None,
    'parent': None,
    'id': None,
    'born': 0,
    'stored': None,
    'emitter': False,
    'evaluated': None,
    'ancestor': None,
    'rew_area': None,
    'gt_bd': None
}
serene_max_emitter_steps = -1
serene_stagnation = 'custom'
serene_archive_selection_operator = 'random'
serene_archive_n_agents2add = 5

