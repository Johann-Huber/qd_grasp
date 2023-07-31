
import random
import utils.constants as consts

from .archive import Archive

class UnstructuredArchive(Archive):

    def __init__(self, archive_limit_size, archive_limit_strat, pop_size, fill_archive_strat, bd_flg):
        super().__init__(fill_archive_strat=fill_archive_strat, bd_flg=bd_flg)

        self._archive_limit_size = archive_limit_size  # max size of the archive
        self._archive_limit_strat = archive_limit_strat  # str defining the strategy to deal with archive size overflow
        self._nb2add_per_update = self._compute_archive_nb2add_per_update(pop_size)  # number of inds to add per update


    def _compute_archive_nb2add_per_update(self, pop_size):
        min_n_inds2add = 1

        n_inds2add_from_pop_pb = consts.PB_ADD2ARCHIVE * pop_size
        n_inds2add = int(n_inds2add_from_pop_pb * consts.OFFSPRING_NB_COEFF)

        return max(n_inds2add, min_n_inds2add)

    def _apply_removal_strategy(self, archive_limit_strat):
        """Applies removal strategy : remove individuals in the archive based on a given strategy."""

        original_len = len(self._inds)
        nb_ind_to_keep = int(original_len * consts.ARCHIVE_DECREMENTAL_RATIO)
        nb_ind_to_remove = original_len - nb_ind_to_keep

        # removal strategies
        if archive_limit_strat == 'random':
            self._apply_removal_strategy_random(nb_ind_to_keep=nb_ind_to_keep)
        else:
            raise NotImplementedError(f'Supported archive_limit_strat : {consts.SUPPORTED_ARCHIVE_LIMIT_STRAT}')

        assert ((original_len - len(self._inds)) == nb_ind_to_remove)

    def _apply_removal_strategy_random(self, nb_ind_to_keep):
        random.shuffle(self._inds)
        self._inds = self._inds[:nb_ind_to_keep]  # remove from grid

    def manage_archive_size(self):
        is_there_removal_strategy = self._archive_limit_size is not None
        if is_there_removal_strategy:
            is_archive_full = len(self._inds) >= self._archive_limit_size
            if is_archive_full:
                self._apply_removal_strategy(archive_limit_strat=self._archive_limit_strat)

    def fill_randomly(self, inds):
        """Randomly fill archive with _archive_nb individuals from pop."""

        fill_arch_count = 0
        already_picked_ind_ids = set()

        inds_added2archive = []

        assert fill_arch_count < self._nb2add_per_update
        is_there_enough_added_inds = False

        while not is_there_enough_added_inds:
            ind_idx = random.randint(0, len(inds) - 1)
            if ind_idx not in already_picked_ind_ids:
                ind = inds[ind_idx]
                self._inds.append(ind)
                inds_added2archive.append(ind)

                fill_arch_count += 1
                already_picked_ind_ids.add(ind_idx)
                is_there_enough_added_inds = fill_arch_count >= self._nb2add_per_update

        return inds_added2archive

