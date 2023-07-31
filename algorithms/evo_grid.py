import pdb

import numpy as np
import utils.common_tools as uct
import utils.constants as consts

class EvoGrid:
    """Grid that discretize the behavior space. Used for monitoring the evo process."""

    def __init__(self, evo_process, bd_filters, measures, nb_cells, bd_bounds):

        # Definitions
        self._measures_flg = measures  # trigger to compute the grids in current run (always skipped otherwise)

        self._grid = None  # grid containing the current BDs which are in the archive
        self._grid_pop_hist = None  # grid_hist containing BDs from all individuals that have been created
        self._grid_pop = None  # grid_pop contains BDs which are in the population

        self._cvt = None  # cvt is the tool to attribute individuals to grid cells (used for both grid and grid_hist)
        self._bd_filters = bd_filters  # filters to extract each BD from the vector of concatenated BDs
        self._evo_process = evo_process  # evolutionary process
        self._nb_cells = nb_cells  # number of cells to discretize behavior spaces
        self._bd_bounds_arr = np.array(bd_bounds)  # bd bounds associated to each behavior spaces

        self._nb_bd = None  # number of behavior descriptors

        # Initializtions
        self._nb_bd = len(bd_filters) if evo_process in consts.MULTI_BD_EVO_PROCESSES else 1

        if measures:
            # initialize the CVT grid # (i) ??????? CVT grid ?
            if evo_process in consts.MULTI_BD_EVO_PROCESSES:
                # grid and cvt will be lists for each BD
                self._grid = []
                self._grid_pop_hist = []
                self._cvt = []
                for bd_filter in self._bd_filters:
                    self._grid.append(np.zeros(nb_cells))  # nb_cells = 1000
                    self._grid_pop_hist.append(np.zeros(nb_cells))
                    cvt_member = uct.CVT(num_centroids=nb_cells, bounds=self._bd_bounds_arr[bd_filter])
                    self._cvt.append(cvt_member)

            else:
                self._grid = np.zeros(nb_cells)
                self._grid_pop_hist = np.zeros(nb_cells)
                self._cvt = uct.CVT(num_centroids=nb_cells, bounds=self._bd_bounds_arr)

    @property
    def nb_cells(self):
        return self._nb_cells

    @property
    def grid(self):
        return self._grid

    @property
    def grid_pop_hist(self):
        return self._grid_pop_hist

    @property
    def grid_pop(self):
        return self._grid_pop

    def add_to_grid(self, pop, target='grid'):
        """Adds a population to grid."""
        valid_targets = ['archive', 'pop_hist', 'pop']
        assert target in valid_targets

        if not self._measures_flg:
            return

        if target == 'archive':
            grid2update = self._grid
        elif target == 'pop_hist':
            grid2update = self._grid_pop_hist
        elif target == 'pop':
            grid2update = self._grid_pop
        else:
            raise RuntimeError(f'Invalid target: {target}.')

        for ind in pop:
            ind_bd = np.array(ind.behavior_descriptor.values)  # np.array([bd1, bd2, bd3, bd4])

            if self._evo_process in consts.MULTI_BD_EVO_PROCESSES:
                # grid and cvt are a list of grids and cvts
                for idx, bd_filter in enumerate(self._bd_filters):
                    bd_value = ind_bd[bd_filter]
                    if not (None in bd_value):  # if the bd has a value
                        grid_index = self._cvt[idx].get_grid_index(bd_value)
                        grid2update[idx][grid_index] += 1
            else:
                grid_index = self._cvt.get_grid_index(ind_bd)
                grid2update[grid_index] += 1

    def _reset_grid_pop(self):
        if self._evo_process in consts.MULTI_BD_EVO_PROCESSES:
            self._grid_pop = []
            for _ in range(self._nb_bd):
                self._grid_pop.append(np.zeros(self._nb_cells))
        else:
            self._grid_pop = np.zeros(self._nb_cells)

    def _update_grid(self, inds_added2archive):
        """Set grid archive with respect to the added individuals population (hist = no reset !)."""
        assert self._measures_flg
        self.add_to_grid(pop=inds_added2archive, target='archive')

    def _update_grid_pop(self, pop):
        """Clean and set grid pop with respect to the current population."""
        assert self._measures_flg
        self._reset_grid_pop()
        self.add_to_grid(pop, target='pop')

    def _update_grid_pop_hist(self, pop):
        """Set grid pop with respect to the current population. No reset."""
        assert self._measures_flg
        self.add_to_grid(pop, target='pop_hist')

    def update_all_grids(self, pop=None, inds_added2archive=None):
        """Update grids with given population's individuals or individuals added to the archive."""
        assert pop is not None or inds_added2archive is not None
        if pop is not None:
            self._update_grid_pop(pop)
            self._update_grid_pop_hist(pop)
        if inds_added2archive is not None:
            self._update_grid(inds_added2archive)

    def reset_grid(self):
        self._grid = np.zeros(self._nb_cells)

    def reset_cvt(self, evaluator):
        self._cvt = uct.CVT(num_centroids=self._nb_cells, bounds=self._bd_bounds_arr)





