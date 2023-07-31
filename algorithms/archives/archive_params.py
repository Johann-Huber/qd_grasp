
from enum import Enum

FillArchiveStrategy = Enum(
    'FillArchiveStrategy',
    ['RANDOM', 'NOVELTY_BASED', 'QD_BASED', 'STRUCTURED_ELITES', 'STRUCTURED_ELITES_WITH_NOVELTY',
     'NONE', 'PYRIBS_ARCHIVE']
)


