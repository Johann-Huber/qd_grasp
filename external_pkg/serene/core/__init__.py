# Created by Giuseppe Paolo 
# Date: 28/07/2020

try:
  from external_pkg.serene.core.evaluator import Evaluator
except:
  print('\033[93m' + "WARNING: Cannot import Evaluator in core/__init__.py" + '\033[0m')