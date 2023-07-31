
import numpy as np

def evaluate_grasp_serene(toolbox, evaluator, inds):
    eval_func = evaluator.ind_evaluation_func

    evaluation_pop = list(toolbox.map(eval_func, inds))

    b_descriptors, is_scs_fitnesses, infos = map(list, zip(*evaluation_pop))

    #objective_batch = np.array(fitnesses)[:, 0].tolist()  # required shape for tell(): (batch_size,)
    measure_batch = b_descriptors
    infos_batch = infos

    objective_batch = [
        ind_info['normalized_multi_fit'] if ind_info['is_success'] else 0. for ind_info in infos
    ]
    #DEBUG----- # TOOO WARNING
    #objective_batch = [
    #    np.random.rand() for ind_info in infos
    #]
    #for i, info in enumerate(infos_batch):
    #    infos_batch[i]['is_success'] = True
    #-------------------
    #pdb.set_trace()

    #print('objective_batch=', objective_batch)
    return objective_batch, measure_batch, infos_batch