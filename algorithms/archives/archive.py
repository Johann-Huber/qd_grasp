import pdb

import numpy as np
import random
import os
from abc import ABC, abstractmethod

from .archive_params import FillArchiveStrategy


class Archive(ABC):
    def __init__(self, fill_archive_strat, bd_flg):
        self._inds = []  # individuals in the archive

        self._bd_flg = bd_flg
        self._fill_archive_strat = fill_archive_strat  # strategy for archive filling
        self._it_export = 0  # how many time did the success archive has been exported

    def __len__(self):
        return len(self._inds)

    @property
    def inds(self):
        return self._inds

    @property
    def fill_archive_strat(self):
        return self._fill_archive_strat

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(f'index must match : 0 <= index < {len(self)} (given: index={index})')

        return self._inds[index]

    def reset(self):
        """Clean the current archive."""
        self._inds = []

    def is_empty(self):
        return len(self._inds) == 0

    def get_successful_inds(self):
        return [ind for ind in self._inds if ind.info.values['is_success']]

    def random_sample(self, n_sample):
        """(Selection function) Randomly sample n_sample individuals from the current population."""

        if n_sample > len(self._inds):
            # concatenation of a list of single ind => sample_ind = [ind] --> sample_ind[0] --> ind
            selected_inds = [random.sample(self._inds, 1)[0] for i_miss in range(n_sample)]
            return selected_inds

        selected_inds = random.sample(self._inds, n_sample)  # already a list of inds

        return selected_inds

    def get_last_id_counter(self):
        if len(self._inds) == 0:
            return -1

        max_id = -1
        for ind in self._inds:
            max_id = max(ind.gen_info.values['id'], max_id)
        return max_id

    def fill_randomly(self, inds):
        raise NotImplementedError('Must be overloaded')

    def fill_novelty_based(self, pop, novelties):
        raise NotImplementedError('Must be overloaded')

    def fill_qd_based(self, inds):
        raise NotImplementedError('Must be overloaded')

    def fill_elites(self, inds):
        raise NotImplementedError('Must be overloaded')

    def fill(self, pop2add, novelties):

        if self._fill_archive_strat == FillArchiveStrategy.RANDOM:
            inds_added2archive = self.fill_randomly(inds=pop2add)

        elif self._fill_archive_strat == FillArchiveStrategy.NOVELTY_BASED:
            assert novelties is not None
            inds_added2archive = self.fill_novelty_based(pop=pop2add, novelties=novelties)

        elif self._fill_archive_strat == FillArchiveStrategy.QD_BASED:
            assert novelties is not None
            inds_added2archive = self.fill_qd_based(inds=pop2add)  #  novelties=novelties

        elif self._fill_archive_strat == FillArchiveStrategy.STRUCTURED_ELITES:
            inds_added2archive = self.fill_elites(inds=pop2add)
            assert inds_added2archive is not None
            # pdb.set_trace() # ok

        elif self._fill_archive_strat == FillArchiveStrategy.STRUCTURED_ELITES_WITH_NOVELTY:
            inds_added2archive = self.fill_elites(inds=pop2add, with_novelty=True)
            assert inds_added2archive is not None
            # pdb.set_trace() # ok

        elif self._fill_archive_strat == FillArchiveStrategy.NONE:
            return []

        else:
            raise NotImplementedError()

        return inds_added2archive

    def export(self, run_name, curr_neval, elapsed_time, verbose=False, only_scs=False):

        export_archive_path = str(run_name) + '/success_archives_qd'
        if not os.path.isdir(export_archive_path):
            os.mkdir(export_archive_path)

        if only_scs:
            ids_inds2export = np.array([i for i, ind in enumerate(self._inds) if ind.info.values['is_success']])
            n_scs_inds = len(ids_inds2export)
            if n_scs_inds > 0:
                inds2export = np.array(self._inds)[ids_inds2export]

                #n_eval_inds = np.array(self._n_eval_inds)[ids_inds2export]
                n_eval_inds = np.array([ind.gen_info['n_eval_generated'] for ind in self._inds])[ids_inds2export]

                bds = np.array([ind.behavior_descriptor.values for ind in self._inds])[ids_inds2export]

                touch_var_fit = np.array(
                    [ind.info.values['touch_var'] for ind in self._inds]
                )[ids_inds2export]

                energy_fit = np.array(
                    [ind.info.values['energy'] for ind in self._inds]
                )[ids_inds2export]

                is_scs_inds = np.array(
                    [ind.info.values['is_success'] for ind in self._inds]
                )[ids_inds2export]
            else:
                inds2export = np.array([])
                n_eval_inds = np.array([])
                bds = np.array([])
                touch_var_fit = np.array([])
                energy_fit = np.array([])
                is_scs_inds = np.array([])

        else:
            inds2export = np.array(self._inds)
            n_eval_inds = np.array([ind.gen_info['n_eval_generated'] for ind in self._inds])
            bds = [ind.behavior_descriptor.values for ind in self._inds]

            touch_var_fit = np.array(
                [ind.info.values['touch_var'] for ind in self._inds]
            )

            energy_fit = np.array(
                [ind.info.values['energy'] for ind in self._inds]
            )

            is_scs_inds = np.array(
                [ind.info.values['is_success'] for ind in self._inds]
            )

        saving = {
            "inds": inds2export,
            "n_eval_inds": n_eval_inds,
            "behavior_descriptors": bds,
            "touch_var_fit": touch_var_fit,
            "energy_fit": energy_fit,
            "is_scs_inds": is_scs_inds,
            "nevals": curr_neval,
            "elapsed_time": elapsed_time
        }

        np.savez_compressed(file=export_archive_path + f'/individuals_{self._it_export}', **saving)

        if verbose:
            print(f'Success archive n{self._it_export} successfully dumped at {export_archive_path}.')

        self._it_export += 1

    @abstractmethod
    def manage_archive_size(self):
        raise RuntimeError('Must be overloaded in subclasses')