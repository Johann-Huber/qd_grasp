import numpy as np
import utils.common_tools as uct
import scipy

import utils.constants as consts
import pdb


class StatsTracker:
    def __init__(self, algo_variant):

        self._outcome_archive_cvg_hist = []  # coverage of As w.r.t. the number of evaluations

        self._success_archive_cvg_hist = []  # coverage of As w.r.t. the number of evaluations
        self._success_archive_qd_score_hist = []  #  sum of the normalized fitness of the archive

        self._outcome_ratio_hist = []  # ratio of evaluated ind that touches the object
        self._success_ratio_hist = []  # ratio of evaluated ind that grasp the object

        self._n_evals = []  # number of evaluations corresponding to each export (each generation)

        self._rolling_n_touched = 0  # number of evaluation considered in update() in which the object is touched
        self._rolling_n_success = 0  # number of evaluation considered in update() in which the object is grasped
        self._rolling_n_rollout = 0  # number of rollouts considered in update()

        self._first_saved_ind_gen = None  # generation at which the first successful ind had been generated
        self._first_saved_ind_n_evals = None  # number of domain evaluation when the 1st scs ind had been generated

        self.algo_variant = algo_variant

    @property
    def first_saved_ind_gen(self):
        return self._first_saved_ind_gen

    @property
    def first_saved_ind_n_evals(self):
        return self._first_saved_ind_n_evals

    def update(self, pop, outcome_archive, curr_n_evals, gen):
        outcome_archive_cvg = self._get_outcome_archive_cvg(outcome_archive)
        success_archive_cvg = self._get_success_archive_cvg(outcome_archive)
        success_archive_qd_score = self._get_success_archive_qd_score(outcome_archive)
        outcome_ratio, success_ratio = self._get_outcome_and_success_ratios(pop, curr_n_evals)

        self._outcome_archive_cvg_hist.append(outcome_archive_cvg)
        self._success_archive_cvg_hist.append(success_archive_cvg)
        self._success_archive_qd_score_hist.append(success_archive_qd_score)
        self._outcome_ratio_hist.append(outcome_ratio)
        self._success_ratio_hist.append(success_ratio)
        self._n_evals.append(curr_n_evals)

    def _get_outcome_archive_sigma_gauss(self, outcome_archive):
        return outcome_archive.get_all_sigma_gauss()

    def _get_outcome_archive_cvg(self, outcome_archive):
        len_Ao = len(outcome_archive)
        max_len_Ao = outcome_archive.get_max_size()
        return len_Ao / max_len_Ao

    def _get_success_archive_cvg(self, outcome_archive):
        max_len_Ao = outcome_archive.get_max_size()
        len_As = outcome_archive.get_n_successful_cells()
        return len_As / max_len_Ao

    def _get_success_archive_qd_score(self, outcome_archive):
        len_As = outcome_archive.get_n_successful_cells()
        if len_As > 0:
            As_inds = outcome_archive.get_successful_inds()
            all_normalized_multi_fit = np.array([ind.info.values['normalized_multi_fit'] for ind in As_inds])
            As_qd_score = all_normalized_multi_fit.sum()
        else:
            As_qd_score = 0
        return As_qd_score

    def _get_outcome_and_success_ratios(self, pop, curr_n_evals):

        pop_n_touched = np.sum([ind.info.values['is_obj_touched'] for ind in pop])
        pop_n_success = np.sum([ind.info.values['is_success'] for ind in pop])
        pop_n_rollout = len(pop)

        self._rolling_n_touched += pop_n_touched
        self._rolling_n_success += pop_n_success
        self._rolling_n_rollout += pop_n_rollout
        assert curr_n_evals == self._rolling_n_rollout  # might cause issues if called twice for a single gen ?

        outcome_ratio = self._rolling_n_touched / self._rolling_n_rollout
        success_ratio = self._rolling_n_success / self._rolling_n_rollout

        return outcome_ratio, success_ratio

    def ending_analysis(self, success_archive):
        pass

    def get_output_data(self):
        output_data_kwargs = {
            'outcome_archive_cvg_hist': np.array(self._outcome_archive_cvg_hist),
            'success_archive_cvg_hist': np.array(self._success_archive_cvg_hist),
            'success_archive_qd_score_hist': np.array(self._success_archive_qd_score_hist),
            'outcome_ratio_hist': np.array(self._outcome_ratio_hist),
            'success_ratio_hist': np.array(self._success_ratio_hist),
            'n_evals_hist': np.array(self._n_evals),
        }

        return output_data_kwargs



