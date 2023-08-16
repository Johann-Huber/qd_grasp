
import pdb


def evaluate_grasp_serene(toolbox, evaluate_fn, inds):

    evaluation_pop = list(toolbox.map(evaluate_fn, inds))

    b_descriptors, is_scs_fitnesses, infos = map(list, zip(*evaluation_pop))
    measure_batch = b_descriptors
    infos_batch = infos

    objective_batch = [
        ind_info['normalized_multi_fit'] if ind_info['is_success'] else 0. for ind_info in infos
    ]
    return objective_batch, measure_batch, infos_batch

