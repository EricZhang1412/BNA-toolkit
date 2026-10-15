# BNA-toolkit

**Brain-Network Analysis toolkit**: a small, well-tested layer over
[bctpy](https://github.com/aestrivex/bctpy) (the Python port of the Brain
Connectivity Toolbox) and complementary libraries, built to be easy to use
and easy to maintain.

Instead of re-implementing graph measures, BNA-toolkit gives them a
consistent interface:

- one container, `BrainNetwork`, that validates the adjacency matrix once and
  carries node labels around;
- measure functions that pick the binary or weighted BCT variant for you and
  return labelled `pandas` objects;
- seeded, reproducible community detection and null models;
- every wrapper cross-checked against `networkx` in the test suite.

## Install

```bash
pip install git+https://github.com/EricZhang1412/BNA-toolkit
# with atlas fetchers, brain plotting and extra null models:
pip install "bna-toolkit[neuro] @ git+https://github.com/EricZhang1412/BNA-toolkit"
```

Python 3.10 to 3.13.

## Quick start

```python
import numpy as np
from bna_toolkit import BrainNetwork, nodal_measures, global_measures, louvain

fc = np.load("connectivity.npy")  # N x N correlation matrix
net = BrainNetwork(fc, labels=region_names)  # diagonal cleared, symmetry detected
net = net.absolute().thresholded(0.15)  # keep strongest 15 % of edges

nodal = nodal_measures(net)  # DataFrame: degree, strength, clustering, ...
summary = global_measures(net)  # Series: density, efficiency, Q, ...
modules, q = louvain(net, seed=0)
```

## What's inside

| Module | Contents |
| --- | --- |
| `bna_toolkit.network` | `BrainNetwork`: validation, thresholding, binarize/normalize, pandas and networkx round-trips |
| `bna_toolkit.measures` | `nodal_measures`, `global_measures` |
| `bna_toolkit.community` | `louvain`, `consensus_louvain` (agreement + consensus clustering) |
| `bna_toolkit.nulls` | `random_nulls` (degree-preserving rewiring), `small_worldness` |

## Design rules

1. Call BCT, don't copy it. Fixes land upstream; this package stays thin.
2. Every public function takes a `BrainNetwork` and returns labelled data.
3. Every randomised routine takes a `seed`.
4. Every wrapper has a test against an independent implementation where one exists.

## Roadmap

- Directed-network support in `measures` and `nulls`
- Rich-club, participation coefficient and within-module z-score helpers
- Group-level utilities: batch measures over subjects, consensus structural networks
- Optional `netneurotools` null models (strength-preserving) behind the `neuro` extra
- Documentation site

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

GPL-3.0-or-later, because the package imports the GPL-3.0 `bctpy`.
Please cite Rubinov & Sporns (2010), *NeuroImage* 52:1059-69 when you use BCT measures.
