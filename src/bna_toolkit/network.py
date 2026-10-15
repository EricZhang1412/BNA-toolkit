"""The :class:`BrainNetwork` container: a validated adjacency matrix with labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import bct
import networkx as nx
import numpy as np
import pandas as pd

Threshold = Literal["absolute", "proportional"]


@dataclass
class BrainNetwork:
    """An undirected or directed brain network stored as a dense adjacency matrix.

    Parameters
    ----------
    adjacency
        Square matrix of connection weights. Values of 0 mean "no edge".
    labels
        Optional node names, one per row. Defaults to ``n0 .. n{N-1}``.
    directed
        If ``None`` (default) the network is treated as undirected when the
        matrix is symmetric, directed otherwise.
    """

    adjacency: np.ndarray
    labels: list[str] = field(default_factory=list)
    directed: bool | None = None

    def __post_init__(self) -> None:
        a = np.asarray(self.adjacency, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"adjacency must be square, got shape {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("adjacency contains NaN or inf; clean the matrix first")
        a = a.copy()
        np.fill_diagonal(a, 0.0)  # BCT functions assume no self-connections
        self.adjacency = a
        if self.directed is None:
            self.directed = not np.allclose(a, a.T)
        if not self.labels:
            self.labels = [f"n{i}" for i in range(a.shape[0])]
        if len(self.labels) != a.shape[0]:
            raise ValueError("labels must have one entry per node")

    # ----------------------------------------------------------------- props
    @property
    def n_nodes(self) -> int:
        return self.adjacency.shape[0]

    @property
    def is_binary(self) -> bool:
        return bool(np.isin(self.adjacency, (0.0, 1.0)).all())

    @property
    def is_signed(self) -> bool:
        return bool((self.adjacency < 0).any())

    @property
    def density(self) -> float:
        n = self.n_nodes
        edges = np.count_nonzero(self.adjacency)
        possible = n * (n - 1)
        if not self.directed:
            edges //= 2
            possible //= 2
        return edges / possible if possible else 0.0

    # ------------------------------------------------------------- builders
    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame, **kwargs) -> BrainNetwork:
        """Build from a square DataFrame whose index holds the node labels."""
        return cls(frame.to_numpy(), labels=[str(i) for i in frame.index], **kwargs)

    @classmethod
    def from_networkx(cls, graph: nx.Graph, weight: str = "weight") -> BrainNetwork:
        nodes = list(graph.nodes)
        a = nx.to_numpy_array(graph, nodelist=nodes, weight=weight)
        return cls(a, labels=[str(n) for n in nodes], directed=graph.is_directed())

    # ------------------------------------------------------------ transforms
    def thresholded(self, value: float, method: Threshold = "proportional") -> BrainNetwork:
        """Keep the strongest edges.

        ``proportional`` keeps the strongest weights up to a *density* of
        ``value`` in (0, 1]: as in BCT, the fraction is taken over all
        possible edges, not over the existing ones, so a network already
        sparser than ``value`` is returned unchanged. ``absolute`` keeps
        weights strictly above ``value``. Both delegate to BCT.
        """
        if method == "proportional":
            if not 0 < value <= 1:
                raise ValueError("proportional threshold must be in (0, 1]")
            w = bct.threshold_proportional(self.adjacency, value)
        elif method == "absolute":
            w = bct.threshold_absolute(self.adjacency, value)
        else:
            raise ValueError(f"unknown threshold method {method!r}")
        return BrainNetwork(w, labels=list(self.labels), directed=self.directed)

    def binarized(self) -> BrainNetwork:
        return BrainNetwork(
            bct.binarize(self.adjacency, copy=True),
            labels=list(self.labels),
            directed=self.directed,
        )

    def normalized(self) -> BrainNetwork:
        """Scale weights to [0, 1] by the maximum absolute weight."""
        w = bct.weight_conversion(self.adjacency, "normalize", copy=True)
        return BrainNetwork(w, labels=list(self.labels), directed=self.directed)

    def absolute(self) -> BrainNetwork:
        return BrainNetwork(
            np.abs(self.adjacency), labels=list(self.labels), directed=self.directed
        )

    # ------------------------------------------------------------- exports
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.adjacency, index=self.labels, columns=self.labels)

    def to_networkx(self) -> nx.Graph:
        cls = nx.DiGraph if self.directed else nx.Graph
        g = nx.from_numpy_array(self.adjacency, create_using=cls)
        return nx.relabel_nodes(g, dict(enumerate(self.labels)))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "directed" if self.directed else "undirected"
        wt = "binary" if self.is_binary else ("signed" if self.is_signed else "weighted")
        return f"BrainNetwork(n_nodes={self.n_nodes}, {kind}, {wt}, density={self.density:.3f})"
