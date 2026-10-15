import networkx as nx
import numpy as np
import pytest

from bna_toolkit import BrainNetwork, global_measures, nodal_measures


def test_nodal_binary_matches_networkx(karate_net, karate):
    nodal = nodal_measures(karate_net)
    assert list(nodal.index) == karate_net.labels
    expected_degree = [d for _, d in karate.degree()]
    np.testing.assert_array_equal(nodal["degree"].to_numpy(), expected_degree)
    np.testing.assert_allclose(nodal["strength"].to_numpy(), expected_degree)

    nx_clust = np.array(list(nx.clustering(karate).values()))
    np.testing.assert_allclose(nodal["clustering"].to_numpy(), nx_clust, atol=1e-12)

    nx_btw = np.array(list(nx.betweenness_centrality(karate, normalized=False).values()))
    corr = np.corrcoef(nodal["betweenness"].to_numpy(), nx_btw)[0, 1]
    assert corr > 0.9999  # BCT counts ordered pairs; proportional to networkx

    nx_eig = np.array(list(nx.eigenvector_centrality_numpy(karate).values()))
    corr = np.corrcoef(nodal["eigenvector_centrality"].to_numpy(), nx_eig)[0, 1]
    assert corr > 0.9999


def test_global_binary_matches_networkx(karate_net, karate):
    g = global_measures(karate_net)
    assert g["n_nodes"] == 34
    assert g["density"] == pytest.approx(nx.density(karate))
    assert g["global_efficiency"] == pytest.approx(nx.global_efficiency(karate))
    assert g["char_path_length"] == pytest.approx(nx.average_shortest_path_length(karate))
    assert g["transitivity"] == pytest.approx(nx.transitivity(karate))
    assert g["mean_clustering"] == pytest.approx(nx.average_clustering(karate))
    assert g["assortativity"] == pytest.approx(
        nx.degree_assortativity_coefficient(karate), abs=1e-6
    )
    assert 0.3 < g["modularity_q"] < 0.5  # karate club Louvain Q is ~0.42


def test_weighted_measures_run(weighted_net):
    nodal = nodal_measures(weighted_net)
    assert nodal.shape == (34, 6)
    assert (nodal["strength"] > 0).all()
    assert (nodal["clustering"] >= 0).all()
    g = global_measures(weighted_net)
    assert 0 < g["global_efficiency"] <= 1
    assert np.isfinite(g["char_path_length"])


def test_signed_input_is_rejected():
    a = np.array([[0, -1.0], [-1.0, 0]])
    with pytest.raises(ValueError, match="non-negative"):
        nodal_measures(BrainNetwork(a))
