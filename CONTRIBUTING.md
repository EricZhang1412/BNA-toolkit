# Contributing

1. Open an issue describing the measure or fix first.
2. Keep wrappers thin: call `bct`, `netneurotools` or `networkx` rather than copying code.
3. Add a test. Compare against `networkx` or the BCT reference data where possible.
4. Run `ruff check .`, `ruff format .` and `pytest` before opening a pull request.
