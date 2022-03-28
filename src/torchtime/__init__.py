from importlib.metadata import version

__version__ = version("torchtime")

from torchtime.collate import sort_by_length  # noqa: F401
from torchtime.data import UEA, PhysioNet2019  # noqa: F401
