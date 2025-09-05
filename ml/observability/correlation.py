"""
Correlation network helpers for observability analyses (off hot-path).

Provides simple primitives used by tests and tooling to analyze connectivity
and prune weak edges.
"""

from __future__ import annotations

from collections.abc import Iterable


def prune_edges(
    edges: Iterable[tuple[str, str, float]],
    *,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """
    Return edges with strength >= threshold.

    Parameters
    ----------
    edges : iterable of (node1, node2, strength)
    threshold : float
        Minimum strength to keep.
    """
    return [(a, b, s) for (a, b, s) in edges if s >= threshold]


def connected_components(nodes: list[str], edges: Iterable[tuple[str, str, float]]) -> int:
    """
    Count connected components in an undirected graph.

    Parameters
    ----------
    nodes : list[str]
        Graph nodes.
    edges : iterable of (node1, node2, strength)
        Graph edges (strength ignored).
    """
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b, _ in edges:
        if a in adj and b in adj and a != b:
            adj[a].add(b)
            adj[b].add(a)
    seen: set[str] = set()
    comps = 0
    for n in nodes:
        if n in seen:
            continue
        # BFS
        q = [n]
        while q:
            cur = q.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            q.extend(adj[cur] - seen)
        comps += 1
    return comps


__all__ = ["connected_components", "prune_edges"]

