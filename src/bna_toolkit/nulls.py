"""Null models and normalised measures (small-worldness)."""

from __future__ import annotations

from collections.abc import Iterator

import bct
import numpy as np

from .network import BrainNetwork


def random_nulls(
    net: BrainNetwork, n: int = 10, swaps_per_edge: int = 10, seed: int | None = 0
) -> Iterator[BrainNetwork]:
    """Yield degree-preserving rewired surrogates (Maslov-Sneppen).

    Uses ``bct.randmio_und_connected`` so each null stays connected when the
    input is connected. Weighted inputs keep their weight distribution.
    """
    if net.directed:
        raise NotImplementedError("random_nulls currently supports undirected networks only")
    rng = np.random.default_rng(seed)
    for _ in range(n):
        r, _eff = bct.randmio_und_connected(
            net.adjacency, swaps_per_edge, seed=int(rng.integers(0, 2**31 - 1))
        )
        yield BrainNetwork(r, labels=list(net.labels), directed=False)


def small_worldness(net: BrainNetwork, n_nulls: int = 20, seed: int | None = 0) -> dict[str, float]:
    """Humphries & Gurney sigma: (C / C_rand) / (L / L_rand).

    Returns a dict with ``clustering``, ``path_length``, their random
    expectations, and ``sigma``. Values above 1 indicate small-world structure.
    """
    if net.directed:
        raise NotImplementedError("small_worldness currently supports undirected networks only")
    w = np.abs(net.adjacency)
    binary = net.is_binary

    def _c_and_l(a: np.ndarray) -> tuple[float, float]:
        if binary:
            c = float(np.mean(bct.clustering_coef_bu(a)))
            d = bct.distance_bin(a)
        else:
            c = float(np.mean(bct.clustering_coef_wu(a)))
            d, _ = bct.distance_wei(bct.weight_conversion(a, "lengths", copy=True))
        length, *_ = bct.charpath(d, include_diagonal=False, include_infinite=False)
        return c, float(length)

    c_obs, l_obs = _c_and_l(w)
    cs, ls = zip(
        *(_c_and_l(null.adjacency) for null in random_nulls(net, n=n_nulls, seed=seed)),
        strict=True,
    )
    c_rand, l_rand = float(np.mean(cs)), float(np.mean(ls))
    gamma = c_obs / c_rand if c_rand else np.nan
    lam = l_obs / l_rand if l_rand else np.nan
    return {
        "clustering": c_obs,
        "clustering_random": c_rand,
        "path_length": l_obs,
        "path_length_random": l_rand,
        "gamma": gamma,
        "lambda": lam,
        "sigma": gamma / lam if lam else np.nan,
    }
