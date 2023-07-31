
from enum import Enum

ControlSpace = Enum('ControlSpace', ['JOINT', 'CARTESIAN'])
GripControlMode = Enum('ControlSpace', ['STANDARD', 'WITH_SYNERGIES'])


NB_ITER_LOCK_BEFORE_START_GRASP = 5

