import pdb
import random
import numpy as np

import utils.constants as consts
from .structured_archive import StructuredArchive


class EliteStructuredArchive(StructuredArchive):
    def __init__(self, fill_archive_strat, bd_flg):
        super().__init__(fill_archive_strat=fill_archive_strat, bd_flg=bd_flg)

        self._targeted_bd_keys = self._init_targeted_bd_keys()

    def _init_targeted_bd_keys(self):
        targeted_bd_keys = consts.BD_FLG_TO_BD_NAMES[self._bd_flg]
        assert isinstance(targeted_bd_keys, list)
        return targeted_bd_keys

    def fill_elites(self, inds):
        """Fill the archive based on given vector of novelties and the fitnesses of individuals."""

        inds_added2archive = []

        all_inds_bds = self._get_bds(inds=inds)

        for ind_candidate, ind_bds in zip(inds, all_inds_bds):

            if not ind_candidate.info.values['is_valid']:
                continue

            targeted_bd_keys = self._targeted_bd_keys
            all_i_bins = [self._get_i_bin_bd(ind_bds, bd_key=bd_key) for bd_key in targeted_bd_keys]
            key_map_ids = tuple(all_i_bins)

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

    def init_qd_archive(self, archive_bootstrap_set):
        inds2add = archive_bootstrap_set.inds
        self.fill_elites(inds=inds2add)

