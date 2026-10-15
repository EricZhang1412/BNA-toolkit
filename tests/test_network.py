import networkx as nx
import numpy as np
import pandas as pd
import pytest

from bna_toolkit import BrainNetwork


def test_basic_properties(karate_net, karate):
    assert karate_net.n_nodes == karate.number_of_nodes()
    assert not karate_net.directed
    assert karate_net.is_binary
    assert not karate_net.is_signed
    assert karate_net.density == pytest.approx(nx.density(karate))


def test_diagonal_is_cleared_and_labels_default():
    a = np.ones((3, 3))
    net = BrainNetwork(a)
    assert np.all(np.diag(net.adjacency) == 0)
    assert net.labels == ["n0", "n1", "n2"]


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="square"):
        BrainNetwork(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="NaN"):
        BrainNetwork(np.array([[0, np.nan], [1, 0]]))
    with pytest.raises(ValueError, match="labels"):
        BrainNetwork(np.zeros((2, 2)), labels=["a"])


def test_directed_detection():
    a = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    assert BrainNetwork(a).directed
    assert not BrainNetwork(a + a.T).directed


def test_proportional_threshold_targets_density(weighted_net):
    n = weighted_net.n_nodes
    thin = weighted_net.thresholded(0.05, "proportional")
    kept = np.count_nonzero(np.triu(thin.adjacency, 1))
    assert kept == round(n * (n - 1) * 0.05 / 2)
    assert thin.density == pytest.approx(0.05, abs=0.005)
    assert thin.labels == weighted_net.labels
    # a network already sparser than the target density is unchanged (BCT semantics)
    same = weighted_net.thresholded(0.5, "proportional")
    np.testing.assert_allclose(same.adjacency, weighted_net.adjacency)


def test_absolute_threshold_and_binarize(weighted_net):
    thin = weighted_net.thresholded(0.5, "absolute")
    assert thin.adjacency[thin.adjacency > 0].min() > 0.5
    assert thin.binarized().is_binary
    with pytest.raises(ValueError):
        weighted_net.thresholded(1.5, "proportional")


def test_normalized_max_is_one(weighted_net):
    assert weighted_net.normalized().adjacency.max() == pytest.approx(1.0)


def test_roundtrips(weighted_net):
    frame = weighted_net.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    back = BrainNetwork.from_dataframe(frame)
    np.testing.assert_allclose(back.adjacency, weighted_net.adjacency)
    assert back.labels == weighted_net.labels

    g = weighted_net.to_networkx()
    assert g.number_of_nodes() == weighted_net.n_nodes
    again = BrainNetwork.from_networkx(g)
    np.testing.assert_allclose(again.adjacency, weighted_net.adjacency)
