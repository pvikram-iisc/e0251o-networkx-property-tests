## E0 251o Project — Property-Based Testing for NetworkX

This repository contains **property-based tests** (Hypothesis) for selected **NetworkX** graph algorithms.

### Algorithms tested
- **Single-source shortest paths (Dijkstra)**: `nx.single_source_dijkstra_path_length`, `nx.single_source_dijkstra`
- **Minimum spanning tree (MST)**: `nx.minimum_spanning_tree`

### What the tests do
The tests automatically generate many graphs (directed/undirected, sparse/dense, weighted/unweighted, edge cases)
and verify mathematical properties including:
- **Invariants** (e.g., source distance is 0; edge-relaxation inequality)
- **Postconditions** (e.g., returned paths match reported distances; MST is a spanning tree with \(n-1\) edges)
- **Metamorphic properties** (e.g., scaling weights scales distances / MST weight)
- **Oracle cross-checks** (e.g., Dijkstra distances match Bellman-Ford for non-negative weights)
- **Brute-force oracle** (MST optimality on small sparse graphs by enumerating all spanning trees)

### How to run

```bash
python -m pip install -r requirements.txt
python -m pytest -q networkx_property_tests.py
```

### Configure how many generated tests run
This test suite uses Hypothesis, which generates many random graphs per test.

You can scale the number of generated examples per test via the `PBT_SCALE` environment variable:

- Default: `PBT_SCALE=1`
- Range: `1` to `50` (values outside are clamped)

Examples:

```powershell
$env:PBT_SCALE=10
python -m pytest -q networkx_property_tests.py
```

```bash
PBT_SCALE=10 python -m pytest -q networkx_property_tests.py
```

### Files
- `networkx_property_tests.py`: the **single required Python file** containing all property-based tests and helpers.

