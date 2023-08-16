
import pdb

import numpy as np
import random
from sklearn.neighbors import NearestNeighbors as Nearest
from pyquaternion import Quaternion

import gym_envs.envs.src.env_constants as env_consts
import utils.constants as consts


def diversity_measure(info):
    grip_or = info['end effector xyzw']
    measure = Quaternion(grip_or[3], grip_or[0], grip_or[1], grip_or[2])
    return measure


def fit_bd_neareast_neighbours(bd_list, nov_metric):
    """Fit kd tree (kNN algo using nov_metric metric) for the given lise of behavior descriptors. Returns the fitted
    kdtree."""
    if len(bd_list) == 0:
        return None

    neigh = Nearest(n_neighbors=consts.K_NN_NOV + 1, metric=nov_metric)
    bds_idx_arr = np.array(bd_list)
    neigh.fit(bds_idx_arr)
    return neigh


def choose_bd_strategy(n_elig_per_bd):
    """Choose the behavior descriptor to use for comparison of novelties in the case of multi_bd novelty search and
    a tournament size >= 2.

    Args:
        inventory (np.array): length of nb_bd, each value counts the number of times the particular bd is evaluated
                              inside the tournament

    Returns:
        int: index of the chosen bd
    """

    #if sum(n_elig_per_bd) == 0:
    #    return None

    # most basic strategy: choose a random behavior descriptor, but make sure inventory(bd_index) > 0
    is_non_empty_bd_found = False
    while not is_non_empty_bd_found:
        chosen_bd_id = random.randint(0, len(n_elig_per_bd) - 1)
        if n_elig_per_bd[chosen_bd_id] > 0:
            is_non_empty_bd_found = True

    return chosen_bd_id


def prepare_n_multi_bd_tournament(pop_size, tournsize_ratio, unwanted_list, putback):
    is_tourn_empty = False

    if tournsize_ratio == 'max':
        if putback:
            tourn_ind_ids = list(
                range(pop_size))  # always pick within the whole pop           # RENAME tourn_idxs AS tourn_ind_ids
        else:
            tourn_ind_ids = [i for i in list(range(pop_size)) if i not in unwanted_list]
            is_tourn_empty = len(tourn_ind_ids) == 0

    else:
        tournsize = int(pop_size * tournsize_ratio)

        if putback:
            tourn_ind_ids = random.sample(range(pop_size), tournsize)
        else:
            list_available_ind_ids = [i for i in list(range(pop_size)) if i not in unwanted_list]
            if len(list_available_ind_ids) < tournsize:
                is_tourn_empty = True
            else:
                random.shuffle(list_available_ind_ids)
                tourn_ind_ids = list_available_ind_ids[:tournsize]

    if is_tourn_empty:
        raise Exception('No enough individuals to generate no putback tournament')

    return tourn_ind_ids


def compute_n_elig_per_bd(pop, tourn_ind_ids, nb_of_bd):
    n_elig_per_bd = np.zeros(nb_of_bd)
    for ind_id in tourn_ind_ids:
        ind = pop[ind_id]
        nov_per_bd_list = list(ind.novelty.values)
        for i_bd, nov_bd in enumerate(nov_per_bd_list):
            if nov_bd is not None:
                n_elig_per_bd[i_bd] += 1

    return n_elig_per_bd


def get_candidate_ind_ids_and_novelties(pop, tourn_ind_ids, bd2compare_id):
    # find all the individuals that are evaluated inside the chosen bd and their novelties
    candidate_ind_ids = []
    candidate_ind_novelties = []

    for ind_id in tourn_ind_ids:
        ind = pop[ind_id]
        nov_list = list(ind.novelty.values)
        try:
            nov_to_compare = nov_list[bd2compare_id]
        except:
            pdb.set_trace()

        if nov_to_compare is not None:
            candidate_ind_ids.append(ind_id)
            candidate_ind_novelties.append(nov_to_compare)

    return candidate_ind_ids, candidate_ind_novelties


def select_n_multi_bd_tournsize(pop, nb_inds2generate, tournsize_ratio, bd_filters, putback=True):

    selected = []
    selected_ind_idx = []

    unwanted_list = []  # in case of no putback
    pop_size = len(pop)
    nb_of_bd = len(bd_filters)

    for i in range(nb_inds2generate):
        # after each iteration, an individual must have been added to selected to get n inds at the end
        tourn_ind_ids = prepare_n_multi_bd_tournament(
            pop_size=pop_size, tournsize_ratio=tournsize_ratio, unwanted_list=unwanted_list, putback=putback
        )

        n_elig_per_bd = compute_n_elig_per_bd(pop=pop, tourn_ind_ids=tourn_ind_ids, nb_of_bd=nb_of_bd)
        if sum(n_elig_per_bd) == 0:
            continue  # no valid bd in the tournament. Skip it.

        bd2compare_id = choose_bd_strategy(n_elig_per_bd=n_elig_per_bd)

        candidate_ind_ids, candidate_ind_novelties = get_candidate_ind_ids_and_novelties(
            pop=pop, tourn_ind_ids=tourn_ind_ids, bd2compare_id=bd2compare_id
        )
        assert len(candidate_ind_novelties) > 0
        max_nov_ind_id = np.argmax(candidate_ind_novelties)

        ind_idx = candidate_ind_ids[max_nov_ind_id]
        selected.append(pop[ind_idx])
        selected_ind_idx.append(ind_idx)
        if not putback:
            unwanted_list.append(ind_idx)

    if len(selected) < nb_inds2generate:
        #  not enough select inds: the pop is filled with randomly sampled solutions
        n_missing_inds = nb_inds2generate - len(selected)
        tourn_ind_ids = [i for i in list(range(pop_size)) if i not in unwanted_list]
        idx_inds2fill = random.sample(tourn_ind_ids, n_missing_inds)
        inds2fill = [pop[idx] for idx in idx_inds2fill]
        selected += inds2fill

    #print('selected_ind_idx=', selected_ind_idx)
    return selected


def compute_bd_filters(bd_indexes):
    """ bd_filters : contains the boolean filters for the different bds"""

    if bd_indexes is None:
        return None

    bd_indexes_arr = np.array(bd_indexes)
    n_bd = len(np.unique(bd_indexes))

    bd_filters = []
    for idx in range(n_bd):
        bd_filters.append(bd_indexes_arr == idx)

    return bd_filters


def get_sigma_gauss_from_ind(ind):
    epsilon = 0.01
    return min(max((ind[-1] + 1) / 2, epsilon), 1)  # project to [0, 1]


def mutate_offspring_qd(off, bound_genotype_thresh, evo_process, mut_flg):

    for ind in off.inds:
        ind.gen_info.values['parent_genome'] = np.array(ind)

    if mut_flg == 'gauss':
        off.var_and(bound_genotype_thresh=bound_genotype_thresh, evo_process=evo_process)
    else:
        raise AttributeError(f'Unknown mutation flag: mut_flg={mut_flg}')


def fill_archive(archive, off, novelties):
    fas = archive.fill_archive_strat

    if fas == 'random':
        inds_added2archive = archive.fill_randomly(pop=off)

    elif fas == 'novelty_based':
        assert novelties is not None
        inds_added2archive = archive.fill_novelty_based(off=off, novelties=novelties)

    elif fas == 'qd_based':
        assert novelties is not None
        inds_added2archive = archive.fill_qd_based(inds=off)

    elif fas == 'structured_archive':
        inds_added2archive = archive.fill_elites(inds=off)
        assert inds_added2archive is not None

    else:
        raise NotImplementedError()

    return inds_added2archive


def replace_pop(pop, ref_pop_inds, evo_process, bd_filters, algo_variant, **kwargs):

    assert algo_variant not in consts.POP_BASED_ALGO_VARIANTS

    # Selection & replacement
    if evo_process == 'ns_rand_multi_bd':
        pop.replace_n_multi_bd_tournsize_based(src_pop_inds=ref_pop_inds, bd_filters=bd_filters,
                                               tournsize_ratio=consts.TOURNSIZE_RATIO,
                                               putback=True)

    elif evo_process == 'random_search':
        pop.replace_random_sample(src_pop=ref_pop_inds)

    elif evo_process in ['ns_nov', 'serene']:
        pop.replace_novelty_based(src_pop=ref_pop_inds)

    elif evo_process == 'fitness':
        pop.replace_fitness_based(src_pop=ref_pop_inds)

    else:
        raise RuntimeError(f"Unknown evo_process: {evo_process}")

    # Update ages
    pop.increment_age()


def is_pareto_efficient(costs, maximise=True):
    """

    Comes from : https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    :param costs: An (n_points, n_costs) array
    :maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient


def get_normalized_multi_fitness(energy_fit, mono_eval_fit, robot_name):
    min_energy_fit_val, max_energy_fit_val = env_consts.ENERGY_FIT_NORM_BOUNDS_PER_ROB[robot_name]['min'], \
                                        env_consts.ENERGY_FIT_NORM_BOUNDS_PER_ROB[robot_name]['max']

    energy_fit_val = np.clip(energy_fit, a_min=min_energy_fit_val, a_max=max_energy_fit_val)
    nrmlzd_energy_fit_val = (energy_fit_val - min_energy_fit_val) / (max_energy_fit_val - min_energy_fit_val)

    min_touch_var_val = consts.FITNESS_TOUCH_VARIANCE_MIN
    max_touch_var_val = consts.FITNESS_TOUCH_VARIANCE_MAX
    mono_eval_fit_val = np.clip(mono_eval_fit, a_min=min_touch_var_val, a_max=max_touch_var_val)
    nrmlzd_mono_eval_fit_val = (mono_eval_fit_val - min_touch_var_val) / (max_touch_var_val - min_touch_var_val)

    normalized_multi_fit = 0.5 * nrmlzd_energy_fit_val + 0.5 * nrmlzd_mono_eval_fit_val
    return normalized_multi_fit


def get_successul_inds_from_set(ind_set):
    return [ind for ind in ind_set if (ind.info.values[consts.SUCCESS_CRITERION])]

