"""Nodal and global graph measures with labelled pandas output.

Every function accepts a :class:`~bna_toolkit.network.BrainNetwork` and
dispatches to the binary/weighted BCT variant automatically.
"""

from __future__ import annotations

import bct
import numpy as np
import pandas as pd

from .network import BrainNetwork


def _check_undirected(net: BrainNetwork, what: str) -> None:
    if net.directed:
        raise NotImplementedError(f"{what} is implemented for undirected networks only (so far)")


def nodal_measures(net: BrainNetwork) -> pd.DataFrame:
    """Per-node measures as a DataFrame indexed by node label.

    Columns: ``degree``, ``strength``, ``clustering``, ``betweenness``,
    ``local_efficiency``, ``eigenvector_centrality``.
    """
    _check_undirected(net, "nodal_measures")
    w = net.adjacency
    binary = net.is_binary
    if net.is_signed:
        raise ValueError("nodal_measures expects non-negative weights; use net.absolute() first")

    out = pd.DataFrame(index=pd.Index(net.labels, name="node"))
    out["degree"] = bct.degrees_und(w)
    out["strength"] = bct.strengths_und(w)
    out["clustering"] = bct.clustering_coef_bu(w) if binary else bct.clustering_coef_wu(w)
    if binary:
        out["betweenness"] = bct.betweenness_bin(w)
        out["local_efficiency"] = bct.efficiency_bin(w, local=True)
    else:
        lengths = bct.weight_conversion(w, "lengths", copy=True)
        out["betweenness"] = bct.betweenness_wei(lengths)
        out["local_efficiency"] = bct.efficiency_wei(w, local=True)
    out["eigenvector_centrality"] = bct.eigenvector_centrality_und(w)
    return out


def global_measures(net: BrainNetwork) -> pd.Series:
    """Whole-network summary as a Series.

    Entries: ``n_nodes``, ``density``, ``mean_degree``, ``mean_strength``,
    ``mean_clustering``, ``transitivity``, ``char_path_length``,
    ``global_efficiency``, ``assortativity``, ``modularity_q``.
    """
    _check_undirected(net, "global_measures")
    if net.is_signed:
        raise ValueError("global_measures expects non-negative weights; use net.absolute() first")
    w = net.adjacency
    binary = net.is_binary

    if binary:
        clustering = bct.clustering_coef_bu(w)
        transitivity = bct.transitivity_bu(w)
        dist = bct.distance_bin(w)
        assortativity = bct.assortativity_bin(w, 0)
    else:
        clustering = bct.clustering_coef_wu(w)
        transitivity = bct.transitivity_wu(w)
        dist, _ = bct.distance_wei(bct.weight_conversion(w, "lengths", copy=True))
        assortativity = bct.assortativity_wei(w, 0)

    char_path, glob_eff, *_ = bct.charpath(dist, include_diagonal=False, include_infinite=False)
    _, q = bct.community_louvain(w, seed=0)

    return pd.Series(
        {
            "n_nodes": net.n_nodes,
            "density": net.density,
            "mean_degree": float(np.mean(bct.degrees_und(w))),
            "mean_strength": float(np.mean(bct.strengths_und(w))),
            "mean_clustering": float(np.mean(clustering)),
            "transitivity": float(transitivity),
            "char_path_length": float(char_path),
            "global_efficiency": float(glob_eff),
            "assortativity": float(assortativity),
            "modularity_q": float(q),
        },
        name="global",
    )
