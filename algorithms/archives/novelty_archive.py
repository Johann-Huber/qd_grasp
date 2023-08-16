import pdb

import numpy as np

import utils.constants as consts
from .unstructured_archive import UnstructuredArchive


class NoveltyArchive(UnstructuredArchive):

    def __init__(self, archive_limit_size, archive_limit_strat, pop_size, fill_archive_strat, bd_flg):

        super().__init__(
            archive_limit_size=archive_limit_size,
            archive_limit_strat=archive_limit_strat,
            pop_size=pop_size,
            fill_archive_strat=fill_archive_strat,
            bd_flg=bd_flg
        )

    def fill_novelty_based(self, pop, novelties):
        """Fill the archive based on given vector of novelties."""

        inds_added2archive = []

        # fill archive with the most novel individuals

        assert len(pop) == len(novelties)

        novel_n = np.array([nov[0] if nov[0] is not None else -consts.INF_FLOAT_CONST for nov in novelties])
        max_novelties_idx = np.argsort(-novel_n)[:self._nb2add_per_update]

        for i in max_novelties_idx:
            if pop.inds[i].behavior_descriptor.values is not None:
                ind = pop.inds[i]
                self._inds.append(ind)
                inds_added2archive.append(ind)

        return inds_added2archive

