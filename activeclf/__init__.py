"""

Active Learning CLassiFier

"""

__version__ = "1.0.0"
__author__ = "Andrea Gardin"


from .datamanager import DataLoader
from .acquisition import DecisionFunction, pointSampler
from .classification import ClassifierModel
from .utils import beauty, misc