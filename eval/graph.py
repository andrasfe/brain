"""Graph traversal — community-structure "rooms" graph, in the spirit of the
CogEval / MAP graph task (arXiv 2310.00194). The paper does not publish the exact
adjacency, so this is a faithful reconstruction: a modular graph with community
structure (Schapiro-style) presented in natural language as connected rooms.

Task (steppath): find the SHORTEST path between two rooms.
Scoring: "% solved" = a valid path (only existing edges) that reaches the goal in
the optimal number of steps; "% invalid" tracks use of non-existent edges.
"""
from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional

# 15-room graph: three communities of 5, ring within each, sparse bridges between
# communities (community structure that prior work found hard for LLMs).
_COMMUNITIES = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
]
_BRIDGES = [(5, 6), (10, 11), (15, 1)]  # ring of communities


def build_graph() -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(1, 16)}
    for comm in _COMMUNITIES:
        for i in range(len(comm)):
            a, b = comm[i], comm[(i + 1) % len(comm)]  # ring within community
            adj[a].add(b)
            adj[b].add(a)
    for a, b in _BRIDGES:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def edges(adj: dict[int, set[int]]) -> list[tuple[int, int]]:
    seen = set()
    out = []
    for a in sorted(adj):
        for b in sorted(adj[a]):
            if (b, a) not in seen:
                seen.add((a, b))
                out.append((a, b))
    return out


def shortest_path(adj: dict[int, set[int]], start: int, goal: int) -> Optional[list[int]]:
    if start == goal:
        return [start]
    seen = {start}
    q = deque([[start]])
    while q:
        path = q.popleft()
        for nxt in sorted(adj[path[-1]]):
            if nxt == goal:
                return path + [nxt]
            if nxt not in seen:
                seen.add(nxt)
                q.append(path + [nxt])
    return None


def optimal_steps(adj, start, goal) -> int:
    p = shortest_path(adj, start, goal)
    return (len(p) - 1) if p else -1


# ── instance generation: pairs with optimal length 2..4 ───────────────────────
@dataclass
class GraphInstance:
    start: int
    goal: int
    optimal: int


def generate_instances(k: int = 10, seed: int = 13) -> list[GraphInstance]:
    adj = build_graph()
    rng = random.Random(seed)
    rooms = list(range(1, 16))
    pairs = []
    for a in rooms:
        for b in rooms:
            if a < b:
                d = optimal_steps(adj, a, b)
                if 2 <= d <= 4:
                    pairs.append((a, b, d))
    rng.shuffle(pairs)
    # spread across distances 2,3,4 for a balanced set
    by_d = {2: [], 3: [], 4: []}
    for a, b, d in pairs:
        by_d[d].append((a, b, d))
    chosen = []
    i = 0
    while len(chosen) < k and any(by_d.values()):
        d = [2, 3, 4][i % 3]
        if by_d[d]:
            a, b, dd = by_d[d].pop()
            chosen.append(GraphInstance(a, b, dd))
        i += 1
    return chosen[:k]


# ── prompts ───────────────────────────────────────────────────────────────────
def _connections_text(adj) -> str:
    return "\n".join(f"Room {a} is connected to room {b}." for a, b in edges(adj))


def zero_shot_prompt(inst: GraphInstance) -> str:
    adj = build_graph()
    return (
        "You are navigating a building of rooms connected by doors. The "
        "connections are:\n\n" + _connections_text(adj) + "\n\n"
        f"Find the shortest path from room {inst.start} to room {inst.goal}. "
        "You may only move between directly connected rooms.\n"
        "Output ONLY the path as a comma-separated list of room numbers, starting "
        f"with {inst.start} and ending with {inst.goal}, e.g.: 3, 5, 6, 11\n"
        "Output nothing else."
    )


def brain_task(inst: GraphInstance) -> str:
    return (
        "Solve this navigation problem by reasoning only (do not write or run "
        "code).\n\n" + zero_shot_prompt(inst)
    )


# ── parsing + scoring ─────────────────────────────────────────────────────────
def parse_path(text: str) -> list[int]:
    """Extract the room sequence: take the longest run of integers on any line."""
    best: list[int] = []
    for line in (text or "").splitlines():
        nums = [int(x) for x in re.findall(r"\d+", line)]
        if len(nums) > len(best):
            best = nums
    if not best:  # fall back to all integers in the text
        best = [int(x) for x in re.findall(r"\d+", text or "")]
    return best


@dataclass
class GraphResult:
    solved: bool
    valid_path: bool
    reached_goal: bool
    n_invalid_edges: int
    length: int
    optimal: int
    parsed_any: bool


def score(inst: GraphInstance, text: str) -> GraphResult:
    adj = build_graph()
    path = parse_path(text)
    if not path:
        return GraphResult(False, False, False, 0, 0, inst.optimal, False)
    n_invalid = 0
    valid = path[0] == inst.start
    for a, b in zip(path, path[1:]):
        if b not in adj.get(a, set()):
            n_invalid += 1
            valid = False
    reached = path[-1] == inst.goal
    length = len(path) - 1
    solved = valid and reached and n_invalid == 0 and length == inst.optimal
    return GraphResult(
        solved=solved, valid_path=valid and n_invalid == 0, reached_goal=reached,
        n_invalid_edges=n_invalid, length=length, optimal=inst.optimal,
        parsed_any=True,
    )
