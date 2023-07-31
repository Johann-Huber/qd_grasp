
import os
import pdb
import numpy as np
import random
import utils.constants as consts
from utils.evo_tools import get_successul_inds_from_set, get_sigma_gauss_from_ind

from .elite_structured_archive import EliteStructuredArchive
from .archive_params import FillArchiveStrategy


class OutcomeArchive(EliteStructuredArchive):
    def __init__(self, bd_flg, fill_archive_strat=FillArchiveStrategy.STRUCTURED_ELITES):
        super().__init__(fill_archive_strat=fill_archive_strat, bd_flg=bd_flg)

        self._it_export_success = 0

    def get_max_size(self):
        # assuming the outcome space is defined by the end effector cartesiean pose
        return self.n_bins_x * self.n_bins_y * self.n_bins_z

    def get_n_successful_cells(self):
        return len(self.get_successful_inds())

    def update(self, pop):
        added_inds, scs_added_inds = self._add_inds(pop=pop)
        return added_inds, scs_added_inds

    def get_all_sigma_gauss(self):
        all_sigma_gauss = [get_sigma_gauss_from_ind(ind) for ind in self.inds]
        all_is_scs = [ind.info.values['is_success'] for ind in self.inds]
        return all_sigma_gauss, all_is_scs

    def _add_inds(self, pop):
        successful_inds = get_successul_inds_from_set(ind_set=pop)
        inds_added2archive = self.fill_elites(pop)
        return inds_added2archive, successful_inds

    def fill_elites(self, inds):
        """Fill the archive based on given vector of novelties and the fitnesses of individuals."""
        # fill_elites overwriting can be avoided by writing the mother method in a more modular way (w.r.t. i_bds)

        inds_added2archive = []

        all_inds_bds = self._get_bds(inds=inds)

        for ind_candidate, ind_bds in zip(inds, all_inds_bds):

            if not ind_candidate.info.values['is_valid']:
                continue

            i_bin_pos_touch_time = self._get_i_bin_bd(ind_bds, bd_key='pos_touch_time')

            key_map_ids = i_bin_pos_touch_time

            is_cell_empty = key_map_ids not in self._map
            if is_cell_empty:
                self._map[key_map_ids] = ind_candidate
                inds_added2archive.append(ind_candidate)

            else:
                # print(f'(ME) not empty niche')
                ind_niche = self._map[key_map_ids]
                candidate_is_scs = ind_candidate.info.values['is_success']
                niche_is_scs = ind_niche.info.values['is_success']

                if not candidate_is_scs and not niche_is_scs:
                    if random.random() > 0.5:
                        self._map[key_map_ids] = ind_candidate
                        inds_added2archive.append(ind_candidate)

                elif not candidate_is_scs and niche_is_scs:
                    pass

                elif candidate_is_scs and not niche_is_scs:
                    self._map[key_map_ids] = ind_candidate
                    inds_added2archive.append(ind_candidate)

                else:
                    assert candidate_is_scs and niche_is_scs
                    candidate_fitness = ind_candidate.info.values['normalized_multi_fit']
                    niche_fitness = ind_niche.info.values['normalized_multi_fit']
                    if candidate_fitness > niche_fitness:
                        # print(f'(ME competition) candidate_fitness={candidate_fitness} | '
                        #      f'niche_fitness={niche_fitness}')
                        self._map[key_map_ids] = ind_candidate
                        inds_added2archive.append(ind_candidate)

        return inds_added2archive


    def export(self, run_name, curr_neval, elapsed_time, verbose=False, only_scs=True):

        inds2export = self.get_successful_inds() if only_scs else self.inds

        export_target_name = 'success_archives' if only_scs else 'outcome_archives'
        export_archive_path = str(run_name) + '/' + export_target_name
        if not os.path.isdir(export_archive_path):
            os.mkdir(export_archive_path)

        bds = [ind.behavior_descriptor.values for ind in inds2export]
        diversity_descriptors = [ind.info.values['diversity_descriptor'] for ind in inds2export]

        n_gen_since_scs_parent_vals = [ind.gen_info.values['n_gen_since_scs_parent'] for ind in inds2export]
        normalized_multi_fit_vals = [ind.info.values['normalized_multi_fit'] for ind in inds2export]

        saving = {
            "inds": np.array(inds2export),
            "behavior_descriptors": bds,
            "fitnesses": normalized_multi_fit_vals,
            "diversity_descriptors": diversity_descriptors,
            "nevals": curr_neval,
            "elapsed_time": elapsed_time,
            "n_gen_since_scs_parent_vals": n_gen_since_scs_parent_vals
        }

        it_export = self._it_export_success if only_scs else self._it_export
        np.savez_compressed(file=export_archive_path + f'/individuals_{it_export}', **saving)

        if verbose:
            print(f'{export_target_name} n°{it_export} has been successfully dumped at {export_archive_path}.')

        if only_scs:
            self._it_export_success += 1
        else:
            self._it_export += 1


