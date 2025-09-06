from __future__ import annotations

from ml.observability.correlation import connected_components
from ml.observability.correlation import prune_edges


def test_prune_edges_and_components() -> None:
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b", 0.9), ("b", "c", 0.4), ("c", "d", 0.8)]
    pruned = prune_edges(edges, threshold=0.5)
    assert len(pruned) == 2
    comps_before = connected_components(nodes, edges)
    comps_after = connected_components(nodes, pruned)
    assert comps_before <= comps_after <= 3
