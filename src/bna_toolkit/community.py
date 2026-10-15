"""Community detection: Louvain with seeds, and consensus clustering.

The agreement and consensus routines are implemented here rather than
delegated to ``bct.agreement`` / ``bct.consensus_und`` because, as of
bctpy 0.6.1, those call ``np.sum`` on a generator, which NumPy 2 rejects.
They follow the same algorithm (Lancichinetti & Fortunato 2012).
"""

from __future__ import annotations

import bct
import numpy as np
import pandas as pd

from .network import BrainNetwork


def louvain(net: BrainNetwork, gamma: float = 1.0, seed: int | None = 0) -> tuple[pd.Series, float]:
    """Run BCT's Louvain once. Returns ``(assignments, Q)``.

    Signed networks are handled with the ``negative_asym`` objective, as the
    BCT documentation recommends for correlation-based functional networks.
    """
    objective = "negative_asym" if net.is_signed else "modularity"
    ci, q = bct.community_louvain(net.adjacency, gamma=gamma, B=objective, seed=seed)
    return pd.Series(ci, index=pd.Index(net.labels, name="node"), name="module"), float(q)


def agreement_matrix(partitions: np.ndarray) -> np.ndarray:
    """Fraction of partitions in which each pair of nodes shares a module.

    ``partitions`` is a ``(n_nodes, n_partitions)`` integer array.
    """
    parts = np.asarray(partitions)
    if parts.ndim != 2:
        raise ValueError("partitions must be a 2-D (n_nodes, n_partitions) array")
    n, m = parts.shape
    agree = np.zeros((n, n), dtype=float)
    for k in range(m):
        col = parts[:, k]
        agree += col[:, None] == col[None, :]
    agree /= m
    np.fill_diagonal(agree, 0.0)
    return agree


def consensus_partition(
    agreement: np.ndarray,
    tau: float = 0.5,
    reps: int = 50,
    gamma: float = 1.0,
    seed: int | None = 0,
    max_iter: int = 50,
) -> np.ndarray:
    """Iterate thresholded-agreement Louvain until all runs agree."""
    rng = np.random.default_rng(seed)
    current = np.array(agreement, dtype=float)
    for _ in range(max_iter):
        current[current < tau] = 0.0
        if not current.any():  # nothing survives the threshold: every node alone
            return np.arange(1, current.shape[0] + 1)
        runs = np.empty((current.shape[0], reps), dtype=int)
        for k in range(reps):
            runs[:, k], _ = bct.community_louvain(
                current, gamma=gamma, seed=int(rng.integers(0, 2**31 - 1))
            )
        current = agreement_matrix(runs)
        off_diag = current[~np.eye(current.shape[0], dtype=bool)]
        if np.all((off_diag == 0.0) | (off_diag == 1.0)):
            return runs[:, 0]
    return runs[:, 0]  # pragma: no cover - did not converge; return the last run


def consensus_louvain(
    net: BrainNetwork,
    n_runs: int = 100,
    gamma: float = 1.0,
    tau: float = 0.5,
    seed: int | None = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Consensus partition over ``n_runs`` Louvain runs.

    Returns the consensus module labels and the node-by-node agreement matrix
    (fraction of runs in which two nodes shared a module).
    """
    rng = np.random.default_rng(seed)
    objective = "negative_asym" if net.is_signed else "modularity"
    partitions = np.empty((net.n_nodes, n_runs), dtype=int)
    for k in range(n_runs):
        partitions[:, k], _ = bct.community_louvain(
            net.adjacency, gamma=gamma, B=objective, seed=int(rng.integers(0, 2**31 - 1))
        )
    agreement = agreement_matrix(partitions)
    consensus = consensus_partition(
        agreement, tau=tau, reps=max(10, n_runs // 10), gamma=gamma, seed=seed
    )
    idx = pd.Index(net.labels, name="node")
    return (
        pd.Series(consensus.astype(int), index=idx, name="module"),
        pd.DataFrame(agreement, index=idx, columns=idx),
    )
