import pdb

import numpy as np
import random
import utils.constants as consts

from .archive import Archive


# to add to consts :
QD_STRUCTURED_ARCHIVE = {
    'approach' : {
        'min_val': -1,
        'max_val': 1,
        'unit_val': None,
    },
    'prehension' : {
        'min_val': -1,
        'max_val': 1,
        'unit_val': None,
    }
}

#------------------------------------------------
# DIMENSIONS FOR CARTESIAN OPERATIONAL SPACE
MIN_X_VAL, MAX_X_VAL = consts.MIN_X_VAL, consts.MAX_X_VAL
MIN_Y_VAL, MAX_Y_VAL = consts.MIN_Y_VAL, consts.MAX_Y_VAL
MIN_Z_VAL, MAX_Z_VAL = consts.MIN_Z_VAL, consts.MAX_Z_VAL

# (0.05 : precision of ~1 cm³) : 2000 cells per submap
N_BINS_PER_DIM_POS_X = 20  # (0.5-(-0.5))/0.05
N_BINS_PER_DIM_POS_Y = 10  # (0.3-(-0.2))/0.05
N_BINS_PER_DIM_POS_Z = 10  # (0.3-(-0.2))/0.05

#------------------------------------------------
# DIMENSIONS FOR ROTATION SAMPLING
MIN_QUATERNION, MAX_QUATERNION = -1, 1
N_BINS_PER_QUATERNION_DIM = 2

#------------------------------------------------
# DIMENSIONS FOR CARTESIAN HAND-OBJECT TOUCH SPACE
MIN_X_TOUCH_VAL, MAX_X_TOUCH_VAL = consts.MIN_X_TOUCH_VAL, consts.MAX_X_TOUCH_VAL
MIN_Y_TOUCH_VAL, MAX_Y_TOUCH_VAL = consts.MIN_Y_TOUCH_VAL, consts.MAX_Y_TOUCH_VAL
MIN_Z_TOUCH_VAL, MAX_Z_TOUCH_VAL = consts.MIN_Z_TOUCH_VAL, consts.MAX_Z_TOUCH_VAL

# (0.02 : precision of 2 cm³) : 1728 cells per submap (a significant part of them cannot be triggered, depending on the object ...
# resulting into an order of magnitude comparable to standard benchmark for QD methods)
#N_BINS_PER_DIM_POS_X_TOUCH = 12  # (0.12-(-0.12))/0.02
#N_BINS_PER_DIM_POS_Y_TOUCH = 12  # (0.25-(0))/0.02
#N_BINS_PER_DIM_POS_Z_TOUCH = 12  # (0.05-(-0.2))/0.02
# trop faible granularité (@100k |As| < 100, @400k |As| < 200)


# (0.01 : precision of 1 cm³) : 15625 cells per submap
N_BINS_PER_DIM_POS_X_TOUCH = 24  # (0.12-(-0.12))/0.01
N_BINS_PER_DIM_POS_Y_TOUCH = 25  # (0.25-(0))/0.01
N_BINS_PER_DIM_POS_Z_TOUCH = 25  # (0.05-(-0.2))/0.01
# plus forte granularité (@100k |As| < 100, @400k |As| < 200)


class StructuredArchive(Archive):
    def __init__(self, fill_archive_strat, bd_flg):
        super().__init__(fill_archive_strat=fill_archive_strat, bd_flg=bd_flg)

        self._bd_flg = bd_flg  # flag associated to behavioral vector

        self.max_x, self.max_y, self.max_z, self.max_q = None, None, None, None  #  bd components max values
        self.min_x, self.min_y, self.min_z, self.min_q = None, None, None, None  #  bd components min values
        self.n_bins_x, self.n_bins_y, self.n_bins_z, self.n_bins_q = None, None, None, None  #  sampling per component
        self.len_bin_x, self.len_bin_y, self.len_bin_z, self.len_bin_q = None, None, None, None  #  bin len per component

        self._archive_type = 'structured_archive'  #  used in fill_archive() for genericity

        self._map = {}
        # main _map :
        # keys = (i_last_obj_pos, i_or_touch_time, i_pos_touch_time, i_grip_or_half, i_grip_pos_half)
        # -> Empty cell : key does not exists in map
        # -> Filled cell : deap.Individual

        self._init_hyperparameters()


    @property
    def archive_type(self):
        return self._archive_type

    @property
    def inds(self):
        return [self._map[key_map_ids] for key_map_ids in self._map]

    def __len__(self):
        return len(self._map)

    def _init_hyperparameters(self):
        self.max_x = MAX_X_VAL
        self.max_y = MAX_Y_VAL
        self.max_z = MAX_Z_VAL
        self.min_x = MIN_X_VAL
        self.min_y = MIN_Y_VAL
        self.min_z = MIN_Z_VAL
        self.max_q = MAX_QUATERNION
        self.min_q = MIN_QUATERNION
        self.max_x_touch = MAX_X_TOUCH_VAL
        self.max_y_touch = MAX_Y_TOUCH_VAL
        self.max_z_touch = MAX_Z_TOUCH_VAL
        self.min_x_touch = MIN_X_TOUCH_VAL
        self.min_y_touch = MIN_Y_TOUCH_VAL
        self.min_z_touch = MIN_Z_TOUCH_VAL

        self.n_bins_x = N_BINS_PER_DIM_POS_X
        self.n_bins_y = N_BINS_PER_DIM_POS_Y
        self.n_bins_z = N_BINS_PER_DIM_POS_Z
        self.n_bins_q = N_BINS_PER_QUATERNION_DIM
        self.n_bins_x_touch = N_BINS_PER_DIM_POS_X_TOUCH
        self.n_bins_y_touch = N_BINS_PER_DIM_POS_Y_TOUCH
        self.n_bins_z_touch = N_BINS_PER_DIM_POS_Z_TOUCH

        self.len_bin_x = (self.max_x - self.min_x) / self.n_bins_x
        self.len_bin_y = (self.max_y - self.min_y) / self.n_bins_y
        self.len_bin_z = (self.max_z - self.min_z) / self.n_bins_z
        self.len_bin_q = (self.max_q - self.min_q) / self.n_bins_q
        self.len_bin_x_touch = (self.max_x_touch - self.min_x_touch) / self.n_bins_x_touch
        self.len_bin_y_touch = (self.max_y_touch - self.min_y_touch) / self.n_bins_y_touch
        self.len_bin_z_touch = (self.max_z_touch - self.min_z_touch) / self.n_bins_z_touch

    def _get_bds(self, inds):
        inds2extract = inds

        if len(inds2extract) == 0:
            return [], []

        if self._bd_flg == 'all_bd':
            all_inds_bds = self._get_bds_all_bd(inds=inds2extract)
        elif self._bd_flg == 'pos_touch':
            all_inds_bds = self._get_bds_pos_touch(inds=inds2extract)
        elif self._bd_flg == 'last_pos_obj_pos_touch':
            all_inds_bds = self._get_bds_last_pos_obj_pos_touch(inds=inds2extract)
        elif self._bd_flg == 'last_pos_obj_pos_touch_pos_half':
            all_inds_bds = self._get_bds_last_pos_obj_pos_touch_pos_half(inds=inds2extract)
        else:
            raise NotImplementedError()

        return all_inds_bds

    def _get_bds_all_bd(self, inds):
        bd_filter_last_obj_pos = 3 * [True] + (4 + 3 + 4 + 3) * [False]
        bd_filter_or_touch_time = 3 * [False] + 4 * [True] + (3 + 4 + 3) * [False]
        bd_filter_pos_touch_time = (3 + 4) * [False] + 3 * [True] + (4 + 3) * [False]
        bd_filter_grip_or_half = (3 + 4 + 3) * [False] + 4 * [True] + 3 * [False]
        bd_filter_grip_pos_half = (3 + 4 + 3 + 4) * [False] + 3 * [True]

        all_inds_bds = []
        for ind in inds:
            bd = ind.behavior_descriptor.values

            bd_last_obj_pos = np.array(bd)[bd_filter_last_obj_pos].tolist()
            bd_or_touch_time = np.array(bd)[bd_filter_or_touch_time].tolist()
            bd_pos_touch_time = np.array(bd)[bd_filter_pos_touch_time].tolist()
            bd_grip_or_half = np.array(bd)[bd_filter_grip_or_half].tolist()
            bd_grip_pos_half = np.array(bd)[bd_filter_grip_pos_half].tolist()

            ind_bds = {
                'last_obj_pos': bd_last_obj_pos,
                'or_touch_time': bd_or_touch_time,
                'pos_touch_time': bd_pos_touch_time,
                'grip_or_half': bd_grip_or_half,
                'grip_pos_half': bd_grip_pos_half,
            }

            all_inds_bds.append(ind_bds)

        return all_inds_bds

    def _get_bds_pos_touch(self, inds):
        bd_filter_pos_touch_time = 3 * [True]

        all_inds_bds = []
        for ind in inds:
            bd = ind.behavior_descriptor.values
            bd_pos_touch_time = np.array(bd)[bd_filter_pos_touch_time].tolist()
            ind_bds = {
                'pos_touch_time': bd_pos_touch_time,
            }
            all_inds_bds.append(ind_bds)
        return all_inds_bds

    def _get_bds_last_pos_obj_pos_touch(self, inds):
        bd_filter_last_obj_pos = 3 * [True] + 3 * [False]
        bd_filter_pos_touch_time = 3 * [False] + 3 * [True]

        all_inds_bds = []
        for ind in inds:
            bd = ind.behavior_descriptor.values
            bd_last_obj_pos = np.array(bd)[bd_filter_last_obj_pos].tolist()
            bd_pos_touch_time = np.array(bd)[bd_filter_pos_touch_time].tolist()
            ind_bds = {
                'last_obj_pos': bd_last_obj_pos,
                'pos_touch_time': bd_pos_touch_time,
            }
            all_inds_bds.append(ind_bds)
        return all_inds_bds


    def _get_bds_last_pos_obj_pos_touch_pos_half(self, inds):
        bd_filter_last_obj_pos = 3 * [True] + 3 * [False] + 3 * [False]
        bd_filter_pos_touch_time = 3 * [False] + 3 * [True] + 3 * [False]
        bd_filter_pos_half_time = 3 * [False] + 3 * [False] + 3 * [True]

        all_inds_bds = []
        for ind in inds:
            bd = ind.behavior_descriptor.values
            bd_last_obj_pos = np.array(bd)[bd_filter_last_obj_pos].tolist()
            bd_pos_touch_time = np.array(bd)[bd_filter_pos_touch_time].tolist()
            bd_pos_half_time = np.array(bd)[bd_filter_pos_half_time].tolist()
            ind_bds = {
                'last_obj_pos': bd_last_obj_pos,
                'pos_touch_time': bd_pos_touch_time,
                'grip_pos_half': bd_pos_half_time,
            }
            all_inds_bds.append(ind_bds)
        return all_inds_bds


    def _get_i_bin_bd(self, ind_bds, bd_key):
        bd_pos_keys = ['last_obj_pos', 'pos_touch_time', 'grip_pos_half']
        bd_or_keys = ['or_touch_time', 'grip_or_half']


        if bd_key not in ind_bds:
            pdb.set_trace()

        if None in ind_bds[bd_key]:
            return None

        if bd_key in bd_pos_keys:
            if bd_key == 'pos_touch_time':
                target_min_x, target_min_y, target_min_z = self.min_x_touch, self.min_y_touch, self.min_z_touch
                target_len_bin_x, target_len_bin_y, target_len_bin_z = \
                    self.len_bin_x_touch, self.len_bin_y_touch, self.len_bin_z_touch
                target_n_bin_x, target_n_bin_y, target_n_bin_z = \
                    self.n_bins_x_touch, self.n_bins_y_touch, self.n_bins_z_touch
            else:
                target_min_x, target_min_y, target_min_z = self.min_x, self.min_y, self.min_z
                target_len_bin_x, target_len_bin_y, target_len_bin_z = self.len_bin_x, self.len_bin_y, self.len_bin_z
                target_n_bin_x, target_n_bin_y, target_n_bin_z = self.n_bins_x, self.n_bins_y, self.n_bins_z

            pos_x, pos_y, pos_z = ind_bds[bd_key]

            i_bin_x = np.floor((pos_x - target_min_x) / target_len_bin_x)
            i_bin_y = np.floor((pos_y - target_min_y) / target_len_bin_y)
            i_bin_z = np.floor((pos_z - target_min_z) / target_len_bin_z)

            # corner cases in which the value is above the specified max -> set to last bin
            i_bin_x = target_n_bin_x - 1 if i_bin_x > target_n_bin_x else i_bin_x
            i_bin_y = target_n_bin_y - 1 if i_bin_y > target_n_bin_y else i_bin_y
            i_bin_z = target_n_bin_z - 1 if i_bin_z > target_n_bin_z else i_bin_z

            i_bin_bd = i_bin_x + i_bin_y * target_n_bin_x + i_bin_z * target_n_bin_x * target_n_bin_y

        else:
            assert bd_key in bd_or_keys

            q1, q2, q3, q4 = ind_bds[bd_key]

            i_bin_q1 = np.floor((q1 - self.min_q) / self.len_bin_q)
            i_bin_q2 = np.floor((q2 - self.min_q) / self.len_bin_q)
            i_bin_q3 = np.floor((q3 - self.min_q) / self.len_bin_q)
            i_bin_q4 = np.floor((q4 - self.min_q) / self.len_bin_q)

            # corner cases in which the value is above the specified max -> set to last bin
            i_bin_q1 = self.n_bins_q - 1 if i_bin_q1 > self.n_bins_q else i_bin_q1
            i_bin_q2 = self.n_bins_q - 1 if i_bin_q2 > self.n_bins_q else i_bin_q2
            i_bin_q3 = self.n_bins_q - 1 if i_bin_q3 > self.n_bins_q else i_bin_q3
            i_bin_q4 = self.n_bins_q - 1 if i_bin_q4 > self.n_bins_q else i_bin_q4

            i_bin_bd = i_bin_q1 + i_bin_q2 * self.n_bins_q + i_bin_q3 * self.n_bins_q ** 2 + \
                       i_bin_q4 * self.n_bins_q ** 3

        return int(i_bin_bd)

    def select_force_success_based(self, n_sample):

        if n_sample > len(self._map):
            # concatenation of a list of single ind
            selected_inds = [self._map[random.sample(list(self._map), 1)[0]] for i_miss in
                             range(n_sample)]
            return selected_inds

        scs_inds = self.get_successful_inds()
        if len(scs_inds) >= n_sample:
            # randomly sample from successes
            selected_inds = random.sample(scs_inds, n_sample)

        elif len(scs_inds) > 0:
            # force filling the pop with successes
            selected_inds = [random.sample(scs_inds, 1)[0] for i_miss in
                             range(n_sample)]

        else:
            # not enough success : fill with randomly sampled non-successful inds
            selected_inds = scs_inds
            n_missing_inds = n_sample - len(selected_inds)

            rand_sel_inds_keys = random.sample(list(self._map), n_missing_inds)
            rand_sel_inds = [self._map[map_ind_key] for map_ind_key in rand_sel_inds_keys]

            selected_inds += rand_sel_inds

        assert len(selected_inds) == n_sample
        return selected_inds


    def select_success_based(self, n_sample):

        if n_sample > len(self._map):
            # concatenation of a list of single ind
            selected_inds = [self._map[random.sample(list(self._map), 1)[0]] for i_miss in
                             range(n_sample)]
            return selected_inds

        scs_inds = self.get_successful_inds()
        if len(scs_inds) >= n_sample:
            # randomly sample from successes
            selected_inds = random.sample(scs_inds, n_sample)
        else:
            # not enough success : fill with randomly sampled non-successful inds
            selected_inds = scs_inds
            n_missing_inds = n_sample - len(selected_inds)

            rand_sel_inds_keys = random.sample(list(self._map), n_missing_inds)
            rand_sel_inds = [self._map[map_ind_key] for map_ind_key in rand_sel_inds_keys]

            selected_inds += rand_sel_inds

        assert len(selected_inds) == n_sample
        return selected_inds

    def select_fitness_based(self, n_sample):

        if n_sample > len(self._map):
            # concatenation of a list of single ind
            selected_inds = [self._map[random.sample(list(self._map), 1)[0]] for i_miss in
                             range(n_sample)]
            return selected_inds

        scs_inds = self.get_successful_inds()
        if len(scs_inds) >= n_sample:
            # sort successes by fitness
            scs_inds_fitnesses = [ind.info.values['normalized_multi_fit'] for ind in scs_inds]
            # select inds with higher fitness
            sorted_scs_ids = np.argsort(scs_inds_fitnesses)[::-1][:n_sample]
            selected_inds = [scs_inds[scs_ind_id] for scs_ind_id in sorted_scs_ids]

        else:
            # not enough success : fill with randomly sampled non-successful inds
            selected_inds = scs_inds
            n_missing_inds = n_sample - len(selected_inds)

            rand_sel_inds_keys = random.sample(list(self._map), n_missing_inds)
            rand_sel_inds = [self._map[map_ind_key] for map_ind_key in rand_sel_inds_keys]

            selected_inds += rand_sel_inds

        assert len(selected_inds) == n_sample
        return selected_inds

    def manage_archive_size(self):
        pass  # empty function for compatibility: structured archive size is bounded by definition

    def reset(self):
        self._map = {}

    def is_empty(self):
        return len(self._map) == 0

    def get_successful_inds(self):
        return [
            self._map[key_map_ids] for key_map_ids in self._map
            if self._map[key_map_ids].info.values[consts.SUCCESS_CRITERION]
        ]

    def random_sample(self, n_sample):
        """(Selection function) Randomly sample n_sample individuals from the current population."""

        if n_sample > len(self.inds):
            # concatenation of a list of single ind => sample_ind = [ind] --> sample_ind[0] --> ind
            selected_inds = [random.sample(self.inds, 1)[0] for i_miss in range(n_sample)]
            return selected_inds

        selected_inds = random.sample(self.inds, n_sample)  # already a list of inds

        return selected_inds