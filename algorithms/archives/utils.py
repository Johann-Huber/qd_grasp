
from .archive_params import FillArchiveStrategy

import utils.constants as consts


def get_fill_archive_strat(evo_process):
    if evo_process in consts.EVO_PROCESS_ARCHIVE_FILL_NOV:
        return FillArchiveStrategy.NOVELTY_BASED
    elif evo_process in consts.EVO_PROCESS_ARCHIVE_LESS:
        return FillArchiveStrategy.NONE
    elif evo_process in consts.EVO_PROCESS_ARCHIVE_STRUCTURED_ELITES:
        return FillArchiveStrategy.STRUCTURED_ELITES
    elif evo_process in consts.CMA_BASED_EVO_PROCESSES:
        return FillArchiveStrategy.PYRIBS_ARCHIVE  # until refactoring
    elif evo_process in consts.SERENE_BASED_EVO_PROCESSES:
        return FillArchiveStrategy.RANDOM  # until refactoring
    else:
        raise NotImplementedError()
        #return FillArchiveStrategy.RANDOM : définir une variable pour éviter les erreurs muettes




