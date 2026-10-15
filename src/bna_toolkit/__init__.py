"""BNA-toolkit: Brain-Network Analysis.

A thin, well-tested layer over ``bctpy`` (the Python port of the Brain
Connectivity Toolbox) and complementary libraries, exposing a consistent
API that returns labelled ``pandas`` objects instead of bare arrays.
"""

from importlib.metadata import PackageNotFoundError, version

from .community import agreement_matrix, consensus_louvain, consensus_partition, louvain
from .measures import global_measures, nodal_measures
from .network import BrainNetwork
from .nulls import random_nulls, small_worldness

try:
    __version__ = version("bna-toolkit")
except PackageNotFoundError:  # editable checkout without metadata
    __version__ = "0.0.0"

__all__ = [
    "BrainNetwork",
    "__version__",
    "agreement_matrix",
    "consensus_partition",
    "consensus_louvain",
    "global_measures",
    "louvain",
    "nodal_measures",
    "random_nulls",
    "small_worldness",
]
