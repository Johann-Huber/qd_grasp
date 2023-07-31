
import random
import utils.constants as consts

from .archive import Archive
from .archive_params import FillArchiveStrategy


class DummyArchive(Archive):
    """Used for archive-less methods, for code consistency. This archive has no effect, but avoid errors while calling
    archive specific methods."""

    def __init__(self, fill_archive_strat=FillArchiveStrategy.NONE, bd_flg=None):
        super().__init__(fill_archive_strat=fill_archive_strat, bd_flg=bd_flg)

    def manage_archive_size(self):
        pass

