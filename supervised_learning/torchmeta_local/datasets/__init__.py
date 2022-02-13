from torchmeta_local.datasets.triplemnist import TripleMNIST
from torchmeta_local.datasets.doublemnist import DoubleMNIST
from torchmeta_local.datasets.cub import CUB
from torchmeta_local.datasets.cifar100 import CIFARFS, FC100
from torchmeta_local.datasets.miniimagenet import MiniImagenet
from torchmeta_local.datasets.omniglot import Omniglot
from torchmeta_local.datasets.tieredimagenet import TieredImagenet
from torchmeta_local.datasets.tcga import TCGA
from torchmeta_local.datasets.pascal5i import Pascal5i
from torchmeta_local.datasets.letter import Letter
from torchmeta_local.datasets.one_hundred_plants_texture import PlantsTexture
from torchmeta_local.datasets.one_hundred_plants_shape import PlantsShape
from torchmeta_local.datasets.one_hundred_plants_margin import PlantsMargin
from torchmeta_local.datasets.bach import Bach

from torchmeta_local.datasets import helpers
from torchmeta_local.datasets import helpers_tabular

__all__ = [
    # image data
    'TCGA',
    'Omniglot',
    'MiniImagenet',
    'TieredImagenet',
    'CIFARFS',
    'FC100',
    'CUB',
    'DoubleMNIST',
    'TripleMNIST',
    'Pascal5i',
    'helpers',
    # tabular data
    'Letter',
    'PlantsTexture',
    'PlantsShape',
    'PlantsMargin',
    'Bach',
    'helpers_tabular'
]
