

import pdb

import numpy as np
from sklearn.neighbors import NearestNeighbors as Nearest
from scipy.spatial import cKDTree as KDTree
from multiprocessing import Pool


from algorithms.population import Population

import utils.constants as consts


def assess_novelties_multi_bd(pop, b_descriptors, bd_indexes, bd_filters, novelty_metric, return_neighbours=False):
    """

    :param pop:
    :param b_descriptors: list of lists containing bds associated to each individuals from the reference pop
                          (pop + archive).
    :param bd_indexes:
    :param bd_filters:
    :param novelty_metric:
    :return:
    """
    novelties = []

    bd_indexes = np.array(bd_indexes)
    nb_bd = len(np.unique(bd_indexes))

    all_pop_bds_are_none, none_novelties = all_bds_none_handler(b_descriptors=b_descriptors, nb_bd=nb_bd, pop=pop)

    if all_pop_bds_are_none:
        undefined_k_tree = None
        return none_novelties, undefined_k_tree

    b_descriptors = np.array(b_descriptors)  # [bd_pop_0, ..., bd_pop_n, bd_archive_0, ..., bd_archive_n]
    bd_lists, tree_ref_pop_indexes = filter_valid_bds(nb_bd=nb_bd, b_descriptors=b_descriptors, bd_filters=bd_filters)

    arg_tuples = [(bd_list, nov_metric) for (bd_list, nov_metric) in zip(bd_lists, novelty_metric)]
    with Pool(40) as p:
        k_trees = p.map(fit_bd_neareast_neighbours_map, arg_tuples)

    novelties = [compute_bd_novelty(bd=b_descriptors[i_pop], bd_filters=bd_filters, k_trees=k_trees) \
                 for i_pop, _ in enumerate(pop)]

    return novelties, k_trees


def compute_average_distance_array(query, k_tree):
    """Finds K nearest neighbours and distances

    Args:
        query (List): behavioral descriptor of individual
        k_tree (Nearest): tree in the behavior descriptor space

    Returns:
        float: average distance to the K nearest neighbours
    """

    n_samples = k_tree.n_samples_fit_
    query = np.array(query)
    if n_samples >= consts.K_NN_NOV + 1:
        neighbours_distances = k_tree.kneighbors(X=query)[0][:, 1:]
    else:
        neighbours_distances = k_tree.kneighbors(X=query, n_neighbors=n_samples)[0][:, 1:]

    avg_distances = np.mean(neighbours_distances, axis=1)
    avg_distance_tuples = [(avg_dist,) for avg_dist in avg_distances]
    return avg_distance_tuples



def assess_novelties_single_bd_vec(pop, reference_pop, novelty_metric, b_descriptors, evo_process='ns_rand_change_bd'):
    novelties = []

    # extract all the behavior descriptors that are not None to create the tree
    reference_pop = [ind for ind in reference_pop if ind.behavior_descriptor.values is not None]  # filter
    b_ds = np.array([ind.behavior_descriptor.values for ind in reference_pop])

    fill_none_val = 0.
    for i_bd, bd in enumerate(b_ds):
        b_ds[i_bd] = np.array([elt if elt is not None else fill_none_val for elt in bd])

    k_tree = Nearest(n_neighbors=consts.K_NN_NOV + 1, metric=novelty_metric)
    k_tree.fit(b_ds)
    # compute novelty for current individuals (loop only on the pop)
    if evo_process == 'ns_rand_change_bd':
        # behavior descriptors can be None
        for i in range(len(pop)):
            if b_descriptors[i] is not None:
                # nov2add = (avg_distance, neighbours_indices)
                nov2add = compute_average_distance(b_descriptors[i], k_tree)
                novelties.append(nov2add)

            else:
                novelties.append((0.0,))
    else:  # compute novelty only
        # nslc variants without normalization
        bd_pop_db = [ind.behavior_descriptor.values for ind in pop]
        try:
            novelties = compute_average_distance_array([ind.behavior_descriptor.values for ind in pop], k_tree)
        except:
            pdb.set_trace()
            pass

    return novelties


def filter_valid_bds(nb_bd, b_descriptors, bd_filters):
    """Filter bds from the np array of reference pop b_descriptors.

    Returns:
        bd_lists: lists of lists containing all valid bds associated to each id_bd;
        tree_ref_pop_indexes: list of lists containing all index in b_descriptors associated to each bds in bd_lists.
    """
    bd_lists = [[] for _ in range(nb_bd)]  # lists of lists containing all valid bds associated to each id_bd
    tree_ref_pop_indexes = [[] for _ in range(nb_bd)]  # list of ref_pop_ids in each tree
    for ref_pop_idx, bd in enumerate(b_descriptors):
        #pdb.set_trace()
        for id_bd, bd_filter in enumerate(bd_filters):
            bd_value = bd[bd_filter]  # extract values associated to the descriptor id_bd

            is_bd_valid = not (None in bd_value)
            if is_bd_valid:
                bd_lists[id_bd].append(bd[bd_filter])
                tree_ref_pop_indexes[id_bd].append(ref_pop_idx)

    return bd_lists, tree_ref_pop_indexes




def all_bds_none_handler(b_descriptors, nb_bd, pop):
    """Compute novelty associated to cases in which all bds are None."""

    all_pop_bds_are_none = True
    for bd in b_descriptors:
        at_least_one_not_none_bd = not all(v is None for v in bd)
        if at_least_one_not_none_bd:
            all_pop_bds_are_none = False
            break

    if all_pop_bds_are_none:
        dummy_none_novel = tuple([None for _ in range(nb_bd)])  # (None, None, ..., None) nb_bd times
        novelties = [dummy_none_novel for _ in range(len(pop))]
    else:
        novelties = None

    return all_pop_bds_are_none, novelties


def compute_bd_novelty(bd, bd_filters, k_trees, return_neighbours=False):

    if return_neighbours:
        novelties_and_neighbours = tuple(
            [compute_average_distance(query=bd[bd_filter], k_tree=k_trees[id_bd], expected_neighbours=True)
             for id_bd, bd_filter in enumerate(bd_filters)]
        )
        try:
            novelties, neighbours = zip(*novelties_and_neighbours)
        except:
            pdb.set_trace()

        return novelties, neighbours

    i_novelty = 0  # output index containing novelty
    novelties = tuple(
        [compute_average_distance(query=bd[bd_filter], k_tree=k_trees[id_bd])[i_novelty] \
         for id_bd, bd_filter in enumerate(bd_filters)]
    )
    return novelties


def compute_average_distance(query, k_tree, expected_neighbours=False):
    """Finds K nearest neighbours and distances

    Args:
        query (List): behavioral descriptor of individual
        k_tree (KDTree or Nearest): tree in the behavior descriptor space

    Returns:
        float: average distance to the K nearest neighbours
        list: indices of the K nearest neighbours
    """
    is_query_invalid = None in query
    if is_query_invalid:
        return (None, None) if expected_neighbours else (None,)  # force consistacy with the code

    if isinstance(k_tree, KDTree):
        raise AttributeError('Depreciated type. Func is still in the code for legacy purpose. Check if necessary.')

    elif isinstance(k_tree, Nearest):
        avg_distance, neighbours_indices = compute_average_distance_nearest(query, k_tree)
    else:
        raise AttributeError(f'Invalid k_tree type: {type(k_tree)} (supported : Nearest)')

    return avg_distance, neighbours_indices


# for debug profiling
def fit_bd_neareast_neighbours_map(tuple_args):
    """Fit kd tree (kNN algo using nov_metric metric) for the given lise of behavior descriptors. Returns the fitted
    kdtree."""
    bd_list, nov_metric = tuple_args
    if len(bd_list) == 0:
        return None

    neigh = Nearest(n_neighbors=consts.K_NN_NOV + 1, metric=nov_metric)
    bds_idx_arr = np.array(bd_list)
    neigh.fit(bds_idx_arr)
    return neigh


def assess_novelties_and_local_quality_single_bd_vec(pop, archive, novelty_metric):

    reference_pop = pop if not archive else pop + archive

    # extract all the behavior descriptors that are not None to create the tree
    reference_pop = [ind for ind in reference_pop if ind.behavior_descriptor.values is not None]  # filter
    b_ds = np.array([ind.behavior_descriptor.values for ind in reference_pop])

    try:
        fit_ref_pop = np.array([ind.info.values['normalized_multi_fit'] for ind in reference_pop])
    except:
        pdb.set_trace()  # not raised: ok

    fill_none_val = 0.
    for i_bd, bd in enumerate(b_ds):
        b_ds[i_bd] = np.array([elt if elt is not None else fill_none_val for elt in bd])

    k_tree = Nearest(n_neighbors=consts.K_NN_NOV + 1, metric=novelty_metric)
    k_tree.fit(b_ds)

    # compute novelty for current individuals (loop only on the pop)
    novelties, n_fit_dominated_neigh = compute_average_distance_and_n_dominated_fit_array(
        [ind.behavior_descriptor.values for ind in pop], k_tree, pop, fit_ref_pop
    )

    local_qualities = n_fit_dominated_neigh
    assert len(novelties) == len(local_qualities)


    return novelties, local_qualities


def compute_average_distance_nearest(query, k_tree):

    n_samples = k_tree.n_samples_fit_
    query = np.array(query)

    n_neigh = consts.K_NN_NOV if n_samples >= consts.K_NN_NOV + 1 else n_samples

    search = k_tree.kneighbors(X=query.reshape(1, -1), n_neighbors=n_neigh)

    neighbours_distances = search[0][0][1:]
    neighbours_indices = search[1][0][1:]

    avg_distance = np.mean(neighbours_distances).item() if len(neighbours_distances) != 0 else consts.INF_NN_DIST

    return avg_distance, neighbours_indices


def assess_novelties(pop, archive, evo_process, bd_indexes, bd_filters, novelty_metric):
    """Compute novelties of current population.

    The function evaluates the novelty for the POPULATION by using both THE POPULATION AND THE ARCHIVE.

    Args:
        pop (list): list of current individuals
        archive (list): list of individuals stored in the archive
        evo_process (str): name of the applied algorithm.
        bd_indexes : None or a list of integer (ex: [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]). (i)
        bd_filters (list): Boolean filters for the different bds
            ex : [array([True, False, False]), array([False, True, False]), array([False, False, True])]

        novelty_metric (list of str, or str): Contains criteria given as argument to Nearest for computing k-means for
        each bd.

    Returns:
        list: list of novelties of current individuals
    """

    reference_pop = pop if not archive else pop + archive  # archive is empty --> only consider current population

    # extract all the behavior descriptors ( [pop[0], ..., pop[n], archive[0], ..., archive[n]] )
    b_descriptors = [ind.behavior_descriptor.values for ind in reference_pop]
    # [bd_pop_0, ..., bd_pop_n, bd_archive_0, ..., bd_archive_n]

    if evo_process in consts.MULTI_BD_EVO_PROCESSES:
        novelties, _ = assess_novelties_multi_bd(
            pop=pop,
            b_descriptors=b_descriptors,
            bd_indexes=bd_indexes,
            bd_filters=bd_filters,
            novelty_metric=novelty_metric)
    else:
        novelties = assess_novelties_single_bd_vec(
            pop=pop,
            reference_pop=reference_pop,
            novelty_metric=novelty_metric,
            evo_process=evo_process,
            b_descriptors=b_descriptors
        )
    return novelties



def compute_average_distance_and_n_dominated_fit_array(query, k_tree, pop, fit_ref_pop):
    """Finds K nearest neighbours and distances

    Args:
        query (List): behavioral descriptor of individual
        k_tree (Nearest): tree in the behavior descriptor space
        pop (List): individuals corresponding to behavior descriptors (queries)

    Returns:
        float: average distance to the K nearest neighbours
    """

    n_samples = k_tree.n_samples_fit_
    query = np.array(query)
    if n_samples >= consts.K_NN_NOV + 1:
        neighbours_distances, neigh_indices = k_tree.kneighbors(X=query, n_neighbors=n_samples)
        neighbours_distances = neighbours_distances[:, 1:]
        neigh_indices = neigh_indices[:, 1:consts.K_NN_LOCAL_QUALITY+1]
        pass
    else:
        neighbours_distances, neigh_indices = k_tree.kneighbors(X=query, n_neighbors=n_samples)
        neighbours_distances = neighbours_distances[:, 1:]
        neigh_indices = neigh_indices[:, 1:consts.K_NN_LOCAL_QUALITY+1]
        pass

    # Compute novelty
    avg_distances = np.mean(neighbours_distances, axis=1)
    avg_distance_tuples = [(avg_dist,) for avg_dist in avg_distances]

    # Compute local quality
    pop_fits = np.array([ind.info.values['normalized_multi_fit'] for ind in pop])
    try:
        neighbours_fits = fit_ref_pop[neigh_indices]
    except:
        pdb.set_trace()

    pop_fits = pop_fits[:, None]
    try:
        assert pop_fits.shape[0] == neighbours_fits.shape[0] and pop_fits.shape[1] == 1 and \
               neighbours_fits.shape[1] <= consts.K_NN_LOCAL_QUALITY
    except:
        pdb.set_trace()

    is_dominating_fit = pop_fits > neighbours_fits
    n_fit_dominated_neigh = np.sum(is_dominating_fit, axis=1)
    assert len(n_fit_dominated_neigh) == len(pop) and len(n_fit_dominated_neigh.shape) == 1

    return avg_distance_tuples, n_fit_dominated_neigh


def update_novelty_routine(is_novelty_required, inds2update, archive, evo_process, bd_indexes, bd_filters,
                           novelty_metric, algo_variant, **kwargs):

    if is_novelty_required:

        if algo_variant in consts.NSLC_ALGO_VARIANTS:
            #  both novelty and local quality must be computed at the same time to avoid multiple KNN calls
            novelties_inds2update, local_qualities_inds2update = \
                assess_novelties_and_local_quality_single_bd_vec(
                    pop=inds2update,
                    archive=archive.inds,
                    novelty_metric=novelty_metric
                )
            Population.update_novelties_inds(inds=inds2update, novelties=novelties_inds2update)
            Population.update_local_qualities_inds(inds=inds2update, local_qualities=local_qualities_inds2update)

        else:
            novelties_inds2update = assess_novelties(
                pop=inds2update,
                archive=archive.inds,
                evo_process=evo_process,
                bd_indexes=bd_indexes,
                bd_filters=bd_filters,
                novelty_metric=novelty_metric
            )
            Population.update_novelties_inds(inds=inds2update, novelties=novelties_inds2update)

    else:
        novelties_inds2update = None

    return novelties_inds2update




