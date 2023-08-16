
import pdb

import argparse
from functools import partial
from pathlib import Path
import gym

from algorithms.controllers.instantiable_controllers import StandardWayPointsJointController, \
    SynergiesWayPointsJointController, StandardWayPointsIKController, SynergiesWayPointsIKController
from utils.common_tools import arg_clean_str

import utils.constants as consts
import gym_envs.envs.src.env_constants as env_consts


def get_controller_class(robot_kwargs):
    controller_type = robot_kwargs['controller_type']

    #controller_type = consts.CONTROLLER
    CONTROLLERS_DICT = {
        'interpolate keypoints speed control grip': StandardWayPointsJointController,
        'interpolate keypoints finger synergies': SynergiesWayPointsJointController,
        'interpolate keypoints cartesian speed control grip': StandardWayPointsIKController,
        'interpolate keypoints cartesian finger synergies': SynergiesWayPointsIKController,
    }

    return CONTROLLERS_DICT[controller_type]


def get_controller_info(robot_kwargs, env_kwargs, algo_variant):
    controller_type = robot_kwargs['controller_type']

    nb_iter = robot_kwargs['nb_iter']
    n_keypoints = consts.NB_KEYPOINTS
    n_genes_per_keypoint = robot_kwargs['gene_per_keypoints']
    n_it_closing_grip = robot_kwargs['n_it_closing_grip']

    #env_params = env_kwargs

    controllers_info_dict = {
        'interpolate keypoints speed control grip': {
            'nb_iter': nb_iter,
            'n_keypoints': n_keypoints,
            'n_genes_per_keypoint': n_genes_per_keypoint,
            'n_it_closing_grip': n_it_closing_grip,
        },

        'interpolate keypoints finger synergies': {
            'nb_iter': nb_iter,
            'n_keypoints': n_keypoints,
            'n_genes_per_keypoint': n_genes_per_keypoint,
            'n_it_closing_grip': n_it_closing_grip,
            'with_synergies': True
        },

        'interpolate keypoints cartesian speed control grip': {
            'nb_iter': nb_iter,
            'n_keypoints': n_keypoints,
            'n_genes_per_keypoint': n_genes_per_keypoint,
            'n_it_closing_grip': n_it_closing_grip,
        },

        'interpolate keypoints cartesian finger synergies': {
            'nb_iter': nb_iter,
            'n_keypoints': n_keypoints,
            'n_genes_per_keypoint': n_genes_per_keypoint,
            'n_it_closing_grip': n_it_closing_grip,
            'with_synergies': True
        },
    }

    controller_info = controllers_info_dict[controller_type]

    controller_info['env_name'] = robot_kwargs['env_name']

    return controller_info



def get_eval_kwargs(parsed_args, robot_kwargs, evo_process, bd_bounds, bd_flg, env_kwargs, prehension_criteria,
                    algo_variant):

    add_iter = int(1*240/robot_kwargs['nb_steps_to_rollout'])
    nb_iter = robot_kwargs['nb_iter']
    bd_bounds = bd_bounds
    nb_steps_to_rollout = robot_kwargs['nb_steps_to_rollout']
    n_it_closing_grip = robot_kwargs['n_it_closing_grip']
    no_contact_table = parsed_args.contact_table

    controller_class = get_controller_class(robot_kwargs)
    controller_info = get_controller_info(robot_kwargs=robot_kwargs, env_kwargs=env_kwargs, algo_variant=algo_variant)


    eval_kwargs = {
        'evo_process': evo_process,
        'bd_flg': bd_flg,
        'prehension_criteria': prehension_criteria,
        'add_iter': add_iter,
        'nb_iter': nb_iter,
        'bd_bounds': bd_bounds,
        'nb_steps_to_rollout': nb_steps_to_rollout,
        'n_it_closing_grip': n_it_closing_grip,
        'no_contact_table': no_contact_table,
        'controller_class': controller_class,
        'controller_info': controller_info,
        'algo_variant': algo_variant,
    }
    return eval_kwargs



def get_robot_kwargs(robot_name, object=None):

    env_name = env_consts.ROBOT_KWARGS[robot_name]['gym_env_name']
    gene_per_keypoints = env_consts.ROBOT_KWARGS[robot_name]['gene_per_keypoints']
    link_id_contact = env_consts.ROBOT_KWARGS[robot_name]['link_id_contact']
    nb_steps_to_rollout = env_consts.ROBOT_KWARGS[robot_name]['nb_steps_to_rollout']

    nb_iter = env_consts.ROBOT_KWARGS[robot_name]['nb_iter_ref']
    nb_iter = int(nb_iter / nb_steps_to_rollout)

    cst_it_close_grip = int(nb_iter * 3 / 4)
    n_it_closing_grip = env_consts.ROBOT_KWARGS[robot_name]['n_it_closing_grip']

    controller_type = env_consts.ROBOT_KWARGS[robot_name]['controller_type']

    robot_kwargs = {
        'env_name': env_name,
        'gene_per_keypoints': gene_per_keypoints,
        'link_id_contact': link_id_contact,
        'nb_steps_to_rollout': nb_steps_to_rollout,
        'nb_iter': nb_iter,
        'cst_it_close_grip': cst_it_close_grip,
        'n_it_closing_grip': n_it_closing_grip,
        'controller_type': controller_type,

    }

    if env_name == 'baxter_grasping-v0':
        grip_slot = env_consts.BX_GRIP_SLOT_PER_OBJECTS[object]
        robot_kwargs['grip_slot'] = grip_slot

    return robot_kwargs


def get_env_kwargs(robot_name, object_name, robot_kwargs):

    env_id = robot_kwargs['env_name']
    steps_to_roll = robot_kwargs['nb_steps_to_rollout']
    fixed_arm = False  # todo depreciated
    controller_type = robot_kwargs['controller_type']

    env_kwargs = {
        'id': env_id,
        'object_name': object_name,
        'steps_to_roll': steps_to_roll,
        'fixed_arm': fixed_arm,
        'controller_type': controller_type,
    }

    if 'grip_slot' in robot_kwargs:
        assert 'baxter' in robot_name, 'grip_slot only defined for baxter'
        env_kwargs['grip_slot'] = robot_kwargs['grip_slot']

    return env_kwargs



def get_genotype_len(robot_kwargs, controller_info):
    """Get the initial genotype length corresponding to CONTROLLER.
    genotype_len will be the amount of parameters to tune for each individuals."""

    with_synergies_controller = controller_info['with_synergies'] if 'with_synergies' in controller_info else False

    n_gens_waypoints = consts.NB_KEYPOINTS * robot_kwargs['gene_per_keypoints']
    n_gen_grasp_primitive_label = 1 if with_synergies_controller else 0

    return n_gens_waypoints + n_gen_grasp_primitive_label


def arg_handler_greater(name, min, value):
    v = int(value)
    if v <= min:
        raise argparse.ArgumentTypeError(f"The {name.strip()} must be greater than {min}")
    return v


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--display",
                        action="store_true",
                        help="Display trajectory generation. Not supported with scoop parallelization.")
    parser.add_argument("-b", "--behavior-descriptor",
                        type=str,
                        default="pos_touch",
                        choices=["pos_touch"],
                        help="Label of the behaviorial vector.")
    parser.add_argument("-r", "--robot",
                        type=arg_clean_str,
                        default="baxter",
                        choices=env_consts.INPUT_ARG_ROBOT2ROBOT_TYPE_NAME.keys(),
                        help="The robot environment")
    parser.add_argument("-o", "--object",
                        type=str,
                        default=None,
                        help="The object to grasp")
    parser.add_argument("-p", "--population",
                        type=partial(arg_handler_greater, "population size", 1),
                        default=100,
                        help="The poulation size")
    parser.add_argument("-g", "--generation",
                        type=partial(arg_handler_greater, "number of generation", 1),
                        default=1000,
                        help="The number of generation")
    parser.add_argument("-x", "--prob-cx",
                       type=float,
                       default=0.,
                       help="Probability to apply crossover.")
    parser.add_argument("-c", "--cells",
                        type=partial(arg_handler_greater, "number of cells", 1),
                        default=1000,
                        help="The number of cells to measure the coverage")
    parser.add_argument("-t", "--contact-table",
                        action="store_true",
                        help="Enable grasp success without touching the table")
    parser.add_argument("-pc", "--prehension-criteria",
                        type=str,
                        default="mono_eval",
                        choices=["mono_eval"],
                        help="Prehension quality criteria.\n" +
                             "\tmono_eval: nrmlizd only mono rollout prehension criteria;\n"
    )
    parser.add_argument("-f", "--folder-name",
                        type=str,
                        default="run",
                        help="Run folder name suffix")
    parser.add_argument("-l", "--log-path",
                        type=str,
                        default=str(Path(__file__).parent.parent / "runs"),
                        help="Run folder name suffix")
    parser.add_argument("-a", "--algorithm",
                        type=arg_clean_str,
                        default="nsmbs",
                        choices=consts.SUPPORTED_VARIANTS_NAMES,
                        help=f"Algorithm variant. Supported variants: {consts.SUPPORTED_VARIANTS_NAMES}")
    parser.add_argument("-e", "--early-stopping",
                        type=int,
                        default=-1,
                        help="Early stopping: the algorithm stops when the number of successes exceed the value")
    parser.add_argument("-s", "--disable-state",
                        action="store_false",
                        help="Disable restore state: load each time the environment (slower) but it is more deterministic")
    parser.add_argument("-nbr", "--n-budget-rollout",
                        type=int,
                        default=None,
                        help="Maximum number of evaluation (= rollout) before ending the evolutionary process.")

    return parser.parse_args()


def get_qd_algo_args(cfg):

    algo_variant = cfg['algorithm']
    pop_size = cfg['evo_proc']['pop_size']
    n_saved_ind_early_stop = cfg['evo_proc']['n_saved_ind_early_stop']
    n_budget_rollouts = cfg['evo_proc']['n_budget_rollouts']
    #bd_flg = cfg['evo_proc']['bd_flg']

    evo_process = cfg['evo_proc']['evo_process']
    mut_flg = cfg['evo_proc']['mut_flg']
    novelty_metric = cfg['evo_proc']['novelty_metric']
    bd_indexes = cfg['evo_proc']['bd_indexes']
    bd_bounds = cfg['evo_proc']['bd_bounds']
    controller_info = cfg['evaluate']['kwargs']['controller_info']
    controller_class = cfg['evaluate']['kwargs']['controller_class']
    prob_cx = cfg['evo_proc']['prob_cx']

    bd_flg = consts.BD_FLG
    bound_genotype_thresh = consts.BOUND_GENOTYPE_THRESH

    obj_vertices_poses = cfg['object']['vertices_poses']
    stabilized_obj_pose = cfg['object']['stabilized_obj_pose']

    robot_name = cfg['robot']['name']
    env_name = cfg['robot']['kwargs']['env_name']
    targeted_object = cfg['env']['kwargs']['object_name']
    controller_type = cfg['robot']['kwargs']['controller_type']
    scene_details = {
        'env_name': env_name,
        'targeted_object': targeted_object,
        'controller_type': controller_type,
    }

    genotype_len = get_genotype_len(robot_kwargs=cfg['robot']['kwargs'], controller_info=controller_info)

    args = {
        'algo_variant': algo_variant,
        'pop_size': pop_size,
        'n_saved_ind_early_stop': n_saved_ind_early_stop,
        'n_budget_rollouts': n_budget_rollouts,
        'bd_flg': bd_flg,
        'evo_process': evo_process,
        'mut_flg': mut_flg,
        'bound_genotype_thresh': bound_genotype_thresh,
        'novelty_metric': novelty_metric,
        'bd_indexes': bd_indexes,
        'bd_bounds': bd_bounds,
        'genotype_len': genotype_len,
        'controller_info': controller_info,
        'controller_class': controller_class,
        'scene_details': scene_details,
        'prob_cx': prob_cx,
        'robot_name': robot_name,
        'obj_vertices_poses': obj_vertices_poses,
        'stabilized_obj_pose': stabilized_obj_pose,
    }

    return args


def get_prehension_criteria_list(prehension_criteria_str):

    if prehension_criteria_str == 'mono_eval':
        return ['touch_var']
    else:
        return AttributeError(f'prehension_criteria_str ({prehension_criteria_str}) must be in \
                SUPPORTED_PREHENSION_CRITERIA_STR={consts.SUPPORTED_PREHENSION_CRITERIA_STR}.')


def initialize_cpu_multicore_data():
    args = parse_input_args()

    DISPLAY = args.display
    POP_SIZE = args.population
    OBJECT = args.object
    PROB_CX = args.prob_cx
    ROBOT = env_consts.INPUT_ARG_ROBOT2ROBOT_TYPE_NAME[args.robot]
    ALGO_VARIANT = args.algorithm
    EVO_PROCESS = consts.ALGO_EVO_PROCESS[ALGO_VARIANT]
    BD_FLG = consts.BD_FLG
    MUT_FLG = consts.ALGO_MUT_FLGS[ALGO_VARIANT]
    BD_BOUNDS = consts.BD_METAPARAMS[BD_FLG]['bd_bounds']
    LOG_PATH = args.log_path
    FOLDER_NAME = args.folder_name
    N_SAVED_IND_EARLY_STOP = args.early_stopping
    N_BUDGET_ROLLOUTS = args.n_budget_rollout
    PREHENSION_CRITERIA_STR = args.prehension_criteria
    assert PREHENSION_CRITERIA_STR in consts.SUPPORTED_PREHENSION_CRITERIA_STR
    PREHENSION_CRITERIA = get_prehension_criteria_list(PREHENSION_CRITERIA_STR)

    ROBOT_KWARGS = get_robot_kwargs(robot_name=ROBOT, object=OBJECT)

    ENV_KWARGS = get_env_kwargs(robot_name=ROBOT, object_name=OBJECT, robot_kwargs=ROBOT_KWARGS)
    ENV_KWARGS['display'] = DISPLAY

    if ALGO_VARIANT in consts.PYRIBS_QD_VARIANTS:
        POP_SIZE = consts.CMA_MAE_POP_SIZE

    assert isinstance(PROB_CX, float)
    assert 0. <= PROB_CX < 1.

    ENV = gym.make(**ENV_KWARGS)  # Initialize env for each parallel worker

    _, obj_vertices_poses = ENV.p.getMeshData(ENV.obj_id)
    stabilized_obj_pose, _ = ENV.p.getBasePositionAndOrientation(ENV.obj_id)

    EVAL_KWARGS = get_eval_kwargs(parsed_args=args,
                                  robot_kwargs=ROBOT_KWARGS,
                                  evo_process=EVO_PROCESS,
                                  bd_flg=BD_FLG,
                                  bd_bounds=BD_BOUNDS,
                                  env_kwargs=ENV_KWARGS,
                                  prehension_criteria=PREHENSION_CRITERIA,
                                  algo_variant=ALGO_VARIANT)

    # assert EVO_PROCESS in consts.MULTI_BD_EVO_PROCESSES
    if EVO_PROCESS in consts.MULTI_BD_EVO_PROCESSES:
        BD_INDEXES = consts.BD_METAPARAMS[BD_FLG]['bd_indexes']
        NOVELTY_METRIC = consts.BD_METAPARAMS[BD_FLG]['novelty_metric']
    else:
        BD_INDEXES = None
        NOVELTY_METRIC = 'minkowski'

    # deal with Scoop parallelization
    '''
    creator.create('BehaviorDescriptor', list)
    creator.create('GenInfo', dict)
    creator.create('Info', dict)
    if EVO_PROCESS in consts.MULTI_BD_EVO_PROCESSES:
        creator.create('Novelty', list)
    else:
        creator.create('Novelty', base.Fitness, weights=(1.0,))

    if EVO_PROCESS in consts.EVO_PROCESS_WITH_LOCAL_COMPETITION:
        creator.create('Fit', base.Fitness, weights=(1.0, 1.0))  #  (novelty, local_quality)
    elif ALGO_VARIANT in consts.ARCHIVE_BASED_NOVELTY_FITNESS_SELECTION_ALGO_VARIANTS:
        creator.create('Fit', base.Fitness, weights=(1.0, 1.0))  #  (novelty, normalized_fitness)
    else:
        creator.create('Fit', base.Fitness, weights=(-1.0,))

    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                   novelty=creator.Novelty, fitness=creator.Fit, info=creator.Info,
                   gen_info=creator.GenInfo)

    evolutionary_process.set_creator(creator)
    '''

    # Scoop requires each core to have the used metaparams loaded in RAM. Those args must therefore be initialized as
    # global variables.

    MULTICORE_SHARED_CFG = {
        'algorithm': ALGO_VARIANT,

        'evo_proc': {
            'pop_size': POP_SIZE,
            'n_saved_ind_early_stop': N_SAVED_IND_EARLY_STOP,
            'n_budget_rollouts': N_BUDGET_ROLLOUTS,
            'bd_flg': BD_FLG,
            'evo_process': EVO_PROCESS,
            'mut_flg': MUT_FLG,
            'novelty_metric': NOVELTY_METRIC,
            'bd_indexes': BD_INDEXES,
            'bd_bounds': BD_BOUNDS,
            'prehension_criteria': PREHENSION_CRITERIA,
            'prob_cx': PROB_CX,

        },

        'env': {
            'kwargs': ENV_KWARGS,
            'initial_state': ENV.get_state(),
        },

        'evaluate': {
            'kwargs': EVAL_KWARGS,
        },

        'robot': {
            'name': ROBOT,
            'kwargs': ROBOT_KWARGS
        },

        'object': {
            'name': OBJECT,
            'vertices_poses': obj_vertices_poses,
            'stabilized_obj_pose': stabilized_obj_pose,
        },

        'output': {
            'log_path': LOG_PATH,
            'folder_name': FOLDER_NAME,
        }

    }

    return ENV, EVAL_KWARGS, MULTICORE_SHARED_CFG


