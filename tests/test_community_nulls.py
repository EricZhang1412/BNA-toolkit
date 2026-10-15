import networkx as nx
import numpy as np

from bna_toolkit import (
    BrainNetwork,
    consensus_louvain,
    louvain,
    random_nulls,
    small_worldness,
)


def test_louvain_is_reproducible(karate_net):
    ci1, q1 = louvain(karate_net, seed=1)
    ci2, q2 = louvain(karate_net, seed=1)
    assert ci1.equals(ci2) and q1 == q2
    assert ci1.nunique() >= 2
    assert list(ci1.index) == karate_net.labels


def test_consensus_shapes(karate_net):
    modules, agreement = consensus_louvain(karate_net, n_runs=20, seed=0)
    assert modules.shape == (34,)
    assert agreement.shape == (34, 34)
    assert np.allclose(agreement.to_numpy(), agreement.to_numpy().T)
    assert agreement.to_numpy().min() >= 0 and agreement.to_numpy().max() <= 1
    assert modules.nunique() >= 2


def test_nulls_preserve_degree(karate_net):
    deg = karate_net.adjacency.sum(1)
    nulls = list(random_nulls(karate_net, n=3, seed=0))
    assert len(nulls) == 3
    for null in nulls:
        np.testing.assert_array_equal(null.adjacency.sum(1), deg)
        assert not np.array_equal(null.adjacency, karate_net.adjacency)


def test_small_world_ring_lattice_rewired():
    g = nx.connected_watts_strogatz_graph(80, 6, 0.1, seed=0)
    res = small_worldness(BrainNetwork.from_networkx(g), n_nulls=5, seed=0)
    assert set(res) >= {"sigma", "gamma", "lambda"}
    assert res["gamma"] > 1.5
    assert res["sigma"] > 1.0


def test_agreement_matrix_basics():
    from bna_toolkit import agreement_matrix

    parts = np.array([[1, 1], [1, 2], [2, 2]])  # 3 nodes, 2 partitions
    agree = agreement_matrix(parts)
    assert agree.shape == (3, 3)
    assert agree[0, 1] == 0.5 and agree[1, 2] == 0.5 and agree[0, 2] == 0.0
    assert np.all(np.diag(agree) == 0)
