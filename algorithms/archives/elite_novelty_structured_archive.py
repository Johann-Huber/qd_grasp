import pdb
import random
import numpy as np

import utils.constants as consts
from .elite_structured_archive import EliteStructuredArchive
from utils.novelty_computation import assess_novelties_single_bd_vec

class EliteNoveltyStructuredArchive(EliteStructuredArchive):
    def __init__(self, fill_archive_strat, bd_flg, novelty_metric):
        super().__init__(fill_archive_strat=fill_archive_strat, bd_flg=bd_flg)

        self._targeted_bd_keys = self._init_targeted_bd_keys()
        self._novelty_metric = novelty_metric

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

        # Novelty must be recomputed for each individual
        self._update_map_novelty()

        return inds_added2archive

    def _update_map_novelty(self):

        # Recompute novelty for each individual in the archive, based on the archive itself
        archive_novelties = assess_novelties_single_bd_vec(
            pop=self.inds,  # individual for which the novelty is about to be computed
            reference_pop=self.inds,  # reference set of inds to compute the novelty
            novelty_metric=self._novelty_metric,
            evo_process='map_elites',
            b_descriptors=None
        )

        assert len(archive_novelties) == len(self.inds)

        # Update novelty for each individual in the archive
        for ind, nov in zip(self.inds, archive_novelties):
            assert len(nov) == 1, 'only works with a single bd vector ; sorting novelty requies a specific method ' \
                                  'otherwise.'
            ind.novelty.values = nov

    def select_novelty_based(self, n_sample):

        if n_sample > len(self._map):
            # concatenation of a list of single ind
            try:
                selected_inds = [self._map[random.sample(list(self._map), 1)[0]] for i_miss in range(n_sample)]
            except Exception as e:
                print('len(self._map)=', len(self._map))
                print('n_sample=', n_sample)
                raise e
            return selected_inds

        # the following novelty sorting stands as long as len(nov) == 1
        all_novelties = [self._map[map_ind_key].novelty.values[0] for map_ind_key in self._map]
        all_key_map_ids = [key_map_id for key_map_id in self._map]

        # select inds with higher novelty
        sorted_key_map_i = np.argsort(all_novelties)[::-1][:n_sample]
        selected_inds = [self._map[all_key_map_ids[map_ind_key_i]] for map_ind_key_i in sorted_key_map_i]

        assert len(selected_inds) == n_sample
        return selected_inds

    def select_novelty_success_based(self, n_sample):

        if n_sample > len(self._map):
            # concatenation of a list of single ind
            selected_inds = [self._map[random.sample(list(self._map), 1)[0]] for i_miss in
                             range(n_sample)]
            return selected_inds

        scs_inds = self.get_successful_inds()
        if len(scs_inds) >= n_sample:
            # sort successes by novelty
            scs_inds_novelties = [ind.novelty.values[0] for ind in scs_inds]  # on récupe bien la nov ici ?

            # select inds with higher novelty
            sorted_scs_ids = np.argsort(scs_inds_novelties)[::-1][:n_sample]
            selected_inds = [scs_inds[scs_ind_id] for scs_ind_id in sorted_scs_ids]

        else:
            # not enough success : fill with randomly sampled non-successful inds
            selected_inds = scs_inds
            n_missing_inds = n_sample - len(selected_inds)

            rand_sel_inds_keys = random.sample(list(self._map), n_missing_inds)
            rand_sel_inds = [self._map[map_ind_key] for map_ind_key in rand_sel_inds_keys]

            selected_inds += rand_sel_inds

        assert len(selected_inds) == n_sample
        return selected_inds


    def select_novelty_fitness_based(self, n_sample, toolbox):

        if n_sample > len(self._map):
            # concatenation of a list of single ind
            selected_inds = [self._map[random.sample(list(self._map), 1)[0]] for i_miss in
                             range(n_sample)]
            return selected_inds

        scs_inds = self.get_successful_inds()
        if len(scs_inds) >= n_sample:

            # set attributes for multi objective selection
            for ind in scs_inds:
                ind.fitness.values = (ind.novelty.values[0], ind.info.values['normalized_multi_fit'])

            # applies NSGA-II
            selected_inds = toolbox.select(scs_inds, k=n_sample)

        else:
            # not enough success : fill with randomly sampled non-successful inds
            selected_inds = scs_inds
            n_missing_inds = n_sample - len(selected_inds)

            rand_sel_inds_keys = random.sample(list(self._map), n_missing_inds)
            rand_sel_inds = [self._map[map_ind_key] for map_ind_key in rand_sel_inds_keys]

            selected_inds += rand_sel_inds

        assert len(selected_inds) == n_sample
        return selected_inds


    def init_qd_archive(self, archive_bootstrap_set):
        inds2add = archive_bootstrap_set.inds
        self.fill_elites(inds=inds2add)

