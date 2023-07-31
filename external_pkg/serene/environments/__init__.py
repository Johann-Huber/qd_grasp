# Created by Giuseppe Paolo 
# Date: 27/07/2020
import sys
import os
from external_pkg.serene.parameters import ROOT_DIR

assets = os.path.join(ROOT_DIR, 'environments/assets/')
sys.path.append(assets)

from external_pkg.serene.environments.environments import registered_envs