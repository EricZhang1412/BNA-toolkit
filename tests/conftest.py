import networkx as nx
import numpy as np
import pytest

from bna_toolkit import BrainNetwork


@pytest.fixture(scope="session")
def karate() -> nx.Graph:
    return nx.karate_club_graph()


@pytest.fixture(scope="session")
def karate_net(karate) -> BrainNetwork:
    return BrainNetwork.from_networkx(karate, weight=None)


@pytest.fixture(scope="session")
def weighted_net(karate) -> BrainNetwork:
    rng = np.random.default_rng(0)
    a = nx.to_numpy_array(karate)
    w = np.triu(rng.uniform(0.1, 1.0, a.shape), 1)
    w = (w + w.T) * a
    return BrainNetwork(w, labels=[f"r{i}" for i in range(a.shape[0])])
