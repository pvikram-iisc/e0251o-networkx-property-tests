"""
E0 251o (2026) Project: Property-Based Testing for NetworkX

Team members: P. Vikram (solo)
Algorithms tested:
- Shortest paths: Dijkstra (single-source)
- Minimum spanning tree (MST)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence, Tuple

import networkx as nx
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


@dataclass(frozen=True)
class WeightedGraphCase:
    """A generated graph plus a chosen source node (when applicable)."""

    G: nx.Graph
    source: int | None


def _edge_weight(G: nx.Graph, u: int, v: int) -> int:
    # NetworkX uses dict-of-dict adjacency; default weight is 1 when absent.
    return int(G[u][v].get("weight", 1))


def _path_weight(G: nx.Graph, path: Sequence[int]) -> int:
    return sum(_edge_weight(G, u, v) for u, v in zip(path, path[1:]))


def _scale_edge_weights(G: nx.Graph, factor: int) -> nx.Graph:
    H = G.__class__()
    H.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        w = int(data.get("weight", 1))
        H.add_edge(u, v, **{**data, "weight": w * factor})
    return H


@st.composite
def weighted_graph_case(
    draw,
    *,
    directed: bool,
    connected: bool = False,
    min_nodes: int = 0,
    max_nodes: int = 14,
    min_weight: int = 0,
    max_weight: int = 20,
    force_unit_weights: bool = False,
) -> WeightedGraphCase:
    """
    Generate a NetworkX Graph/DiGraph with integer non-negative weights.

    Generation highlights:
    - Varying node counts and densities.
    - Optional enforced connectivity (used for MST tests on undirected graphs).
    - Non-negative weights to satisfy Dijkstra's preconditions.
    """

    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = list(range(n))
    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)

    if n <= 1:
        source = draw(st.none() if n == 0 else st.just(0))
        return WeightedGraphCase(G=G, source=source)

    existing_edges: set[Tuple[int, int]] = set()

    if connected:
        assume(not directed)  # connectivity enforcement is only used for undirected MST cases
        # Create a random spanning tree first to guarantee connectivity.
        for i in range(1, n):
            parent = draw(st.integers(min_value=0, max_value=i - 1))
            u, v = parent, i
            existing_edges.add((min(u, v), max(u, v)))

    if directed:
        all_pairs = [(u, v) for u in nodes for v in nodes if u != v]
        base_edges = set()  # directed graphs: store as ordered pairs
    else:
        all_pairs = [(u, v) for u in nodes for v in nodes if u < v]
        base_edges = set(existing_edges)

    remaining = [e for e in all_pairs if e not in base_edges]
    # Choose a random subset of remaining edges to vary density.
    extra_edges = draw(
        st.sets(st.sampled_from(remaining), max_size=min(len(remaining), n * (n - 1) // 2))
        if remaining
        else st.just(set())
    )

    edges: Iterable[Tuple[int, int]]
    if directed:
        edges = list(extra_edges)
    else:
        edges = list(base_edges | extra_edges)

    for u, v in edges:
        w = 1 if force_unit_weights else draw(st.integers(min_value=min_weight, max_value=max_weight))
        G.add_edge(u, v, weight=int(w))

    source = draw(st.integers(min_value=0, max_value=n - 1))
    return WeightedGraphCase(G=G, source=source)


def any_weighted_graph_case(**kwargs):
    """
    Build a strategy that explores both directed and undirected graphs.

    Hypothesis strategies must be composed explicitly; passing a strategy object into a
    boolean parameter would be truthy and defeat exploration.
    """

    return st.booleans().flatmap(lambda d: weighted_graph_case(directed=d, **kwargs))


@st.composite
def small_sparse_connected_weighted_graph(
    draw,
    *,
    min_nodes: int = 2,
    max_nodes: int = 7,
    extra_edges_max: int = 3,
    min_weight: int = 0,
    max_weight: int = 30,
) -> nx.Graph:
    """
    Generate a *small* connected undirected weighted graph with few extra edges.

    Purpose: enable brute-force verification of MST optimality by enumerating all spanning trees.
    By limiting edge count to (n-1 + extra_edges_max), the number of candidate spanning trees
    remains small enough to check inside a property test.
    """

    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = list(range(n))
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Start from a random spanning tree backbone to guarantee connectivity.
    tree_edges: list[Tuple[int, int]] = []
    used: set[Tuple[int, int]] = set()
    for i in range(1, n):
        parent = draw(st.integers(min_value=0, max_value=i - 1))
        u, v = parent, i
        e = (min(u, v), max(u, v))
        used.add(e)
        tree_edges.append(e)

    # Add a small number of extra edges to create cycles.
    all_pairs = [(u, v) for u in nodes for v in nodes if u < v]
    remaining = [e for e in all_pairs if e not in used]
    extra_edges = draw(st.sets(st.sampled_from(remaining), max_size=min(extra_edges_max, len(remaining))) if remaining else st.just(set()))

    for u, v in list(tree_edges) + list(extra_edges):
        w = draw(st.integers(min_value=min_weight, max_value=max_weight))
        G.add_edge(u, v, weight=int(w))

    assume(nx.is_connected(G))
    return G


# ----------------------------
# Dijkstra (single-source) tests
# ----------------------------


@given(st.booleans())
@settings(max_examples=50, deadline=None)
def test_dijkstra_empty_graph_raises_node_not_found(directed: bool):
    """
    Property: Running Dijkstra with a source node not present in the graph raises `NodeNotFound`.

    Mathematical basis: The single-source shortest path problem is defined with respect to a
    source vertex in the graph. If the source vertex does not exist, the problem instance is
    ill-formed and should be rejected rather than producing arbitrary output.

    Test strategy: Generate empty graphs as part of the general graph generator; for
    the empty-graph case, call the function with a source that cannot exist and verify that
    NetworkX raises the expected exception.

    Assumptions: None (the graph has zero nodes, so any integer source is absent).

    Bug signal: If no exception is raised, it suggests missing input validation and may hide
    incorrect behavior on boundary cases.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    with pytest.raises(nx.NodeNotFound):
        nx.single_source_dijkstra_path_length(G, source=0, weight="weight")


@given(weighted_graph_case(directed=True, connected=False))
@settings(max_examples=200, deadline=None)
def test_dijkstra_source_distance_is_zero_for_existing_source(case: WeightedGraphCase):
    """
    Property: For a non-empty graph, the shortest-path distance from the source to itself is 0.

    Mathematical basis: The empty walk from a node to itself has total weight 0. With
    non-negative weights, no path can have negative total weight, so 0 is minimal.

    Test strategy: Generate directed weighted graphs with non-negative weights, pick a
    source node, and verify that the returned distance for the source is exactly 0.

    Assumptions: The source exists in the graph; weights are non-negative (Dijkstra precondition).

    Bug signal: If the source distance is missing or non-zero, the algorithm's initialization
    or relaxation logic is incorrect.
    """

    assume(case.G.number_of_nodes() > 0)
    assert case.source is not None

    dist = nx.single_source_dijkstra_path_length(case.G, source=case.source, weight="weight")
    assert dist[case.source] == 0


@given(any_weighted_graph_case(connected=False))
@settings(max_examples=150, deadline=None)
def test_dijkstra_edge_relaxation_upper_bounds_hold(case: WeightedGraphCase):
    """
    Property: Dijkstra distances satisfy the edge-relaxation inequality:
    for every edge (u -> v) with weight w, if u is reachable then dist[v] <= dist[u] + w.

    Mathematical basis: The shortest distance dist[v] is the minimum over all path lengths
    from source to v. Any path reaching u can be extended by edge (u, v) to form a valid
    candidate path to v with length dist[u] + w, so dist[v] cannot exceed that candidate.

    Test strategy: Generate (di)graphs with non-negative weights, compute single-source
    distances, and check the inequality across all edges where dist[u] is defined.

    Assumptions: Non-negative weights. For undirected graphs, each edge is treated as
    two directed arcs by NetworkX's adjacency iteration.

    Bug signal: A violation implies computed distances are not globally minimal, indicating
    incorrect relaxation ordering or update logic.
    """

    assume(case.G.number_of_nodes() > 0)
    assert case.source is not None

    dist = nx.single_source_dijkstra_path_length(case.G, source=case.source, weight="weight")
    for u, v, data in case.G.edges(data=True):
        w = int(data.get("weight", 1))
        if u in dist and v in dist:
            assert dist[v] <= dist[u] + w
        if (not case.G.is_directed()) and u in dist and v in dist:
            # For undirected graphs the same inequality also holds in the opposite direction.
            assert dist[u] <= dist[v] + w


@given(any_weighted_graph_case(connected=False))
@settings(max_examples=120, deadline=None)
def test_dijkstra_paths_match_reported_distances(case: WeightedGraphCase):
    """
    Property: The paths returned by Dijkstra are valid and their total weights match the reported distances.

    Mathematical basis: For each reachable target t, the algorithm returns a path P(source, t)
    with length equal to the computed shortest distance dist[t]. This is a postcondition that
    ensures internal consistency of the output pair (dist, path).

    Test strategy: Run `nx.single_source_dijkstra` to obtain both distances and paths. For each
    reachable node, verify:
    - path starts at source and ends at target
    - consecutive nodes are edges in the graph
    - sum of edge weights along the path equals dist[target]

    Assumptions: Non-negative weights. The graph may be disconnected; only reachable nodes appear.

    Bug signal: Mismatches indicate an inconsistency between the predecessor structure and the
    distance estimates, suggesting a reconstruction or bookkeeping bug.
    """

    assume(case.G.number_of_nodes() > 0)
    assert case.source is not None

    dist, paths = nx.single_source_dijkstra(case.G, source=case.source, weight="weight")
    for t, p in paths.items():
        assert p[0] == case.source
        assert p[-1] == t
        for u, v in zip(p, p[1:]):
            assert case.G.has_edge(u, v)
        assert _path_weight(case.G, p) == dist[t]


@given(
    any_weighted_graph_case(connected=False),
    st.integers(min_value=1, max_value=9),
)
@settings(max_examples=100, deadline=None)
def test_dijkstra_weight_scaling_scales_distances(case: WeightedGraphCase, k: int):
    """
    Metamorphic property: Multiplying all edge weights by a positive factor k scales all shortest-path
    distances by k, for the same source.

    Mathematical basis: For any path P, its total weight scales by k. Therefore the minimum over all
    scaled path weights equals k times the minimum over original path weights: dist_k(t) = k * dist(t).

    Test strategy: Generate graphs with non-negative weights, compute distances from a source, create
    a new graph with weights multiplied by k, and verify all reachable distances scale exactly.

    Assumptions: k > 0; weights are integers and non-negative; Dijkstra's conditions remain valid.

    Bug signal: A violation implies distances depend on absolute weight magnitudes in an invalid way
    (e.g., incorrect arithmetic, overflow-like behavior, or mishandling of weight attributes).
    """

    assume(case.G.number_of_nodes() > 0)
    assert case.source is not None

    dist1 = nx.single_source_dijkstra_path_length(case.G, source=case.source, weight="weight")
    G2 = _scale_edge_weights(case.G, k)
    dist2 = nx.single_source_dijkstra_path_length(G2, source=case.source, weight="weight")

    assert dist2.keys() == dist1.keys()
    for t in dist1:
        assert dist2[t] == k * dist1[t]


@given(any_weighted_graph_case(connected=False, force_unit_weights=True))
@settings(max_examples=120, deadline=None)
def test_unit_weight_dijkstra_matches_unweighted_shortest_paths(case: WeightedGraphCase):
    """
    Metamorphic/consistency property: When every edge weight is 1, Dijkstra distances equal
    unweighted shortest-path distances (minimum number of edges).

    Mathematical basis: With unit weights, path weight equals hop count. Minimizing total weight
    is equivalent to minimizing the number of edges, which is exactly the unweighted shortest-path
    problem solved by BFS on unweighted graphs.

    Test strategy: Generate graphs where all edges have weight 1. Compare
    `single_source_dijkstra_path_length` with `single_source_shortest_path_length` from the same source.

    Assumptions: All edge weights are 1 (enforced by generation). Graph may be directed or undirected.

    Bug signal: Disagreement indicates Dijkstra mishandles uniform weights or the weight attribute.
    """

    assume(case.G.number_of_nodes() > 0)
    assert case.source is not None

    dist_weighted = nx.single_source_dijkstra_path_length(case.G, source=case.source, weight="weight")
    dist_unweighted = nx.single_source_shortest_path_length(case.G, source=case.source)
    assert dist_weighted == dist_unweighted


@given(any_weighted_graph_case(connected=False, max_nodes=12, min_weight=0, max_weight=25))
@settings(max_examples=120, deadline=None)
def test_dijkstra_matches_bellman_ford_on_nonnegative_weights(case: WeightedGraphCase):
    """
    Oracle property: On graphs with non-negative weights, Dijkstra and Bellman-Ford must agree.

    Mathematical basis: Both algorithms compute the single-source shortest path distances in the
    same weighted graph model. Dijkstra is correct under non-negative weights; Bellman-Ford is
    correct even with negative weights (when no negative cycles). Therefore, on non-negative weights,
    they must compute identical distances for every reachable node.

    Test strategy: Generate weighted (di)graphs with non-negative integer weights, compute distances
    from a random source using:
    - `nx.single_source_dijkstra_path_length`
    - `nx.single_source_bellman_ford_path_length`
    and compare the resulting dictionaries.

    Assumptions: All edge weights are non-negative (enforced by generator). Source exists.

    Bug signal: Any disagreement is strong evidence of a correctness bug in at least one of the
    algorithms or in the weight-handling logic.
    """

    assume(case.G.number_of_nodes() > 0)
    assert case.source is not None

    d1 = nx.single_source_dijkstra_path_length(case.G, source=case.source, weight="weight")
    d2 = nx.single_source_bellman_ford_path_length(case.G, source=case.source, weight="weight")
    assert d1 == d2


# ----------------------------
# Minimum spanning tree (MST) tests
# ----------------------------


@given(weighted_graph_case(directed=False, connected=True, min_nodes=1, max_nodes=16, min_weight=0, max_weight=30))
@settings(max_examples=120, deadline=None)
def test_mst_is_tree_and_spans_all_nodes(case: WeightedGraphCase):
    """
    Property: For a connected undirected graph G with n >= 1 nodes, the MST returned by NetworkX
    is a spanning tree: it is connected, acyclic, and includes all original nodes.

    Mathematical basis: A spanning tree is, by definition, an acyclic connected subgraph that spans
    all vertices. Any minimum spanning tree must be a spanning tree.

    Test strategy: Generate random connected undirected weighted graphs (via an enforced random
    spanning-tree backbone plus extra edges). Compute `nx.minimum_spanning_tree(G)` and assert:
    - node set matches G's node set
    - the result is a tree (`nx.is_tree`)

    Assumptions: Graph is connected and undirected.

    Bug signal: If the result is not a tree or misses nodes, the algorithm violates MST postconditions.
    """

    G = case.G
    assume(G.number_of_nodes() >= 1)
    assume(nx.is_connected(G))

    T = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    assert set(T.nodes()) == set(G.nodes())
    assert nx.is_tree(T)


@given(weighted_graph_case(directed=False, connected=True, min_nodes=1, max_nodes=18, min_weight=0, max_weight=30))
@settings(max_examples=120, deadline=None)
def test_mst_edge_count_is_n_minus_1(case: WeightedGraphCase):
    """
    Property: A spanning tree on n nodes has exactly n-1 edges; therefore an MST must have n-1 edges.

    Mathematical basis: A fundamental theorem about trees: any tree with n vertices has n-1 edges.
    Conversely, any connected acyclic graph on n vertices must have n-1 edges.

    Test strategy: Generate connected undirected weighted graphs and compute an MST. Verify the MST
    edge count equals n-1.

    Assumptions: The input graph is connected and has n >= 1.

    Bug signal: Too few edges means it is not spanning/connected; too many implies a cycle.
    """

    G = case.G
    assume(G.number_of_nodes() >= 1)
    assume(nx.is_connected(G))

    T = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    assert T.number_of_edges() == G.number_of_nodes() - 1


def _mst_total_weight(T: nx.Graph) -> int:
    return sum(int(data.get("weight", 1)) for _, _, data in T.edges(data=True))


@given(weighted_graph_case(directed=False, connected=True, min_nodes=2, max_nodes=16, min_weight=0, max_weight=40))
@settings(max_examples=90, deadline=None)
def test_mst_cycle_property_holds_for_non_tree_edges(case: WeightedGraphCase):
    """
    Property (cycle / exchange characterization): Let T be an MST of G. For any non-tree edge e=(u,v),
    let P be the unique path between u and v in T. Then the maximum edge weight on P is <= w(e).

    Mathematical basis: If there were an edge e with w(e) smaller than the maximum-weight edge on the
    cycle formed by adding e to T, then swapping would produce a strictly lighter spanning tree,
    contradicting minimality. With ties, the non-strict inequality still holds for some MST, and
    should hold for the MST returned by the algorithm.

    Test strategy: Generate connected undirected weighted graphs, compute an MST T, then for each
    edge e in G \\ T:
    - find the path P between e's endpoints in T
    - compute the maximum weight edge along P
    - assert max(P) <= w(e)

    Assumptions: Input graph is connected (so T has unique paths); undirected; weights are comparable.

    Bug signal: A violation indicates the produced tree is not minimal, i.e., not an MST.
    """

    G = case.G
    assume(nx.is_connected(G))

    T = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    tree_edges = {tuple(sorted((u, v))) for u, v in T.edges()}

    for u, v, data in G.edges(data=True):
        e = tuple(sorted((u, v)))
        if e in tree_edges:
            continue
        w_e = int(data.get("weight", 1))
        path = nx.shortest_path(T, source=u, target=v)  # unique in a tree
        max_on_path = 0
        for a, b in zip(path, path[1:]):
            max_on_path = max(max_on_path, _edge_weight(T, a, b))
        assert max_on_path <= w_e


@given(
    weighted_graph_case(directed=False, connected=True, min_nodes=1, max_nodes=16, min_weight=0, max_weight=40),
    st.integers(min_value=1, max_value=9),
)
@settings(max_examples=80, deadline=None)
def test_mst_total_weight_scales_with_weight_scaling(case: WeightedGraphCase, k: int):
    """
    Metamorphic property: If all edge weights are multiplied by a positive factor k, the total
    weight of an MST scales by k.

    Mathematical basis: Every spanning tree's total weight scales by k under uniform scaling, so the
    minimum total weight over all spanning trees scales by k as well.

    Test strategy: Generate connected weighted graphs, compute MST total weight W, scale all edge
    weights by k, compute a new MST, and verify its total weight equals k*W.

    Assumptions: k > 0; graph is connected and undirected.

    Bug signal: A violation indicates the algorithm is not minimizing the correct objective or is
    mishandling weights.
    """

    G = case.G
    assume(nx.is_connected(G))

    T1 = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    W1 = _mst_total_weight(T1)

    G2 = _scale_edge_weights(G, k)
    T2 = nx.minimum_spanning_tree(G2, weight="weight", algorithm="kruskal")
    W2 = _mst_total_weight(T2)

    assert W2 == k * W1


@given(small_sparse_connected_weighted_graph())
@settings(max_examples=120, deadline=None)
def test_mst_is_optimal_by_bruteforce_on_small_sparse_graphs(G: nx.Graph):
    """
    Oracle property (brute-force optimality): For small connected graphs, the MST returned by NetworkX
    has total weight equal to the minimum weight over *all* spanning trees.

    Mathematical basis: By definition, an MST minimizes total edge weight among all spanning trees.
    For small graphs we can exhaustively enumerate every candidate spanning tree and compute its weight,
    providing a definitive oracle (not just a necessary condition).

    Test strategy:
    - Generate a connected undirected graph with n in [2, 7] and only a few extra edges.
    - Compute NetworkX's MST weight.
    - Enumerate all subsets of edges of size n-1; keep those that form a spanning tree; compute their
      total weights and take the minimum.
    - Assert equality with NetworkX's MST weight.

    Assumptions: Graph is connected and undirected. Edge weights are integers.

    Bug signal: A mismatch is high-quality evidence that the returned structure is not truly minimal.
    Hypothesis will shrink the failing graph to a minimal counterexample, which is ideal bonus evidence.
    """

    assume(G.number_of_nodes() >= 2)
    assume(nx.is_connected(G))

    T = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    w_mst = _mst_total_weight(T)

    edges = list(G.edges(data=True))
    n = G.number_of_nodes()
    best = None
    for subset in combinations(edges, n - 1):
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from([(u, v, data) for (u, v, data) in subset])
        if nx.is_tree(H):
            w = sum(int(data.get("weight", 1)) for _, _, data in subset)
            best = w if best is None else min(best, w)

    assert best is not None
    assert w_mst == best

