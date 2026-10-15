"""Minimal end-to-end example on a synthetic small-world network."""

import networkx as nx

from bna_toolkit import BrainNetwork, global_measures, louvain, nodal_measures, small_worldness

g = nx.connected_watts_strogatz_graph(60, 6, 0.1, seed=0)
net = BrainNetwork.from_networkx(g)
print(net)
print(global_measures(net).round(3))
print(nodal_measures(net).head())
modules, q = louvain(net)
print(f"Louvain found {modules.nunique()} modules, Q = {q:.3f}")
print({k: round(v, 3) for k, v in small_worldness(net, n_nulls=5).items()})
