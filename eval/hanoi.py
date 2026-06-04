"""Tower of Hanoi — text isomorph, faithful to MAP paper (arXiv 2310.00194).

State: three lists A, B, C holding integers 0..n-1.
Lists are written left-to-right = bottom-to-top; the rightmost element is the
movable "top of stack". Reachable states have each list in ascending order.

Rules (verbatim intent from the paper):
  R1: You can only move a number if it is at the rightmost end of its list.
  R2: You can only move a number to the rightmost end of a list if it is larger
      than the other numbers in that list.

Goal: A=[], B=[], C=[0,1,...,n-1]  (all numbers in C, ascending).
"""
from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

PEGS = ("A", "B", "C")


# ── state helpers ─────────────────────────────────────────────────────────────
State = dict  # {"A": [..], "B": [..], "C": [..]}


def goal_state(n: int) -> State:
    return {"A": [], "B": [], "C": list(range(n))}


def clone(s: State) -> State:
    return {k: list(v) for k, v in s.items()}


def state_key(s: State) -> tuple:
    return (tuple(s["A"]), tuple(s["B"]), tuple(s["C"]))


def render_state(s: State) -> str:
    return f"A = {s['A']}\nB = {s['B']}\nC = {s['C']}"


# ── move validity ─────────────────────────────────────────────────────────────
def is_valid_move(s: State, n: int, src: str, trg: str) -> bool:
    if src not in PEGS or trg not in PEGS or src == trg:
        return False
    if not s[src] or s[src][-1] != n:      # R1: n must be the rightmost of src
        return False
    if s[trg] and n < s[trg][-1]:          # R2: n must exceed target's top
        return False
    return True


def apply_move(s: State, n: int, src: str, trg: str) -> State:
    s = clone(s)
    s[src].pop()
    s[trg].append(n)
    return s


# ── optimal solver (BFS over the tiny state space) ────────────────────────────
def optimal_length(initial: State, n: int) -> int:
    goal = state_key(goal_state(n))
    start = state_key(initial)
    if start == goal:
        return 0
    seen = {start}
    q = deque([(initial, 0)])
    while q:
        s, d = q.popleft()
        for src in PEGS:
            if not s[src]:
                continue
            num = s[src][-1]
            for trg in PEGS:
                if src == trg:
                    continue
                if is_valid_move(s, num, src, trg):
                    ns = apply_move(s, num, src, trg)
                    k = state_key(ns)
                    if k == goal:
                        return d + 1
                    if k not in seen:
                        seen.add(k)
                        q.append((ns, d + 1))
    return -1  # unreachable (shouldn't happen for valid ToH states)


# ── instance generation ───────────────────────────────────────────────────────
def all_reachable_states(n: int) -> list[State]:
    """Every assignment of disks to pegs, each peg sorted ascending."""
    states = []
    for assign in product(PEGS, repeat=n):
        s = {"A": [], "B": [], "C": []}
        for disk, peg in enumerate(assign):
            s[peg].append(disk)
        for p in PEGS:
            s[p].sort()
        states.append(s)
    return states


# The two states the paper holds out as Actor in-context examples.
_ICL_HOLDOUT = {
    (("0", "1"), ("2",), ()),   # A=[0,1] B=[2] C=[]  (rendered as ints below)
    (("1",), ("0",), ("2",)),   # A=[1]   B=[0] C=[2]
}


def _holdout_keys() -> set:
    return {((0, 1), (2,), ()), ((1,), (0,), (2,))}


def generate_instances(n: int = 3, k: int = 10, seed: int = 7) -> list[State]:
    """Sample k distinct non-goal initial states (excludes ICL holdouts for n=3)."""
    rng = random.Random(seed)
    goal = state_key(goal_state(n))
    holdout = _holdout_keys() if n == 3 else set()
    pool = [s for s in all_reachable_states(n)
            if state_key(s) != goal and state_key(s) not in holdout]
    rng.shuffle(pool)
    # prefer instances needing at least 2 moves (non-trivial)
    nontrivial = [s for s in pool if optimal_length(s, n) >= 2]
    chosen = (nontrivial or pool)[:k]
    return chosen


# ── prompts (faithful to the paper's zero-shot format) ────────────────────────
RULES_HEADER = (
    "Consider the following puzzle problem:\n\n"
    "Problem description:\n"
    "- There are three lists labeled A, B, and C.\n"
    "- There is a set of numbers distributed among those three lists.\n"
    "- You can only move numbers from the rightmost end of one list to the "
    "rightmost end of another list.\n\n"
    "Rule #1: You can only move a number if it is at the rightmost end of its current list.\n"
    "Rule #2: You can only move a number to the rightmost end of a list if it is "
    "larger than the other numbers in that list.\n"
    "A move is valid if it satisfies both Rule #1 and Rule #2.\n"
    "A move is invalid if it violates either Rule #1 or Rule #2.\n"
)


def zero_shot_prompt(initial: State, n: int) -> str:
    goal = goal_state(n)
    return (
        RULES_HEADER
        + "\nGoal: The goal is to end up in the configuration where all numbers "
        "are in list C, in ascending order using minimum number of moves.\n\n"
        "This is the starting configuration:\n" + render_state(initial) + "\n"
        "This is the goal configuration:\n" + render_state(goal) + "\n"
        "Give me the sequence of moves to solve the puzzle from the starting "
        "configuration. Please try to use as few moves as possible, and make sure "
        "to follow the rules listed above. Please limit your answer to a maximum "
        f"of 10 steps.\nFormat each move EXACTLY as:\nMove <N> from <src> to <tgt>.\n"
        "Output only the move lines, one per line, and nothing else."
    )


def brain_task(initial: State, n: int) -> str:
    """Task text handed to the multi-agent brain (it must PLAN, not run code)."""
    return (
        "Solve this Tower-of-Hanoi-style puzzle by reasoning only (do not write or "
        "run code).\n\n" + zero_shot_prompt(initial, n)
    )


# ── output parsing + scoring ──────────────────────────────────────────────────
_MOVE_RE = re.compile(r"move\s+(\d+)\s+from\s+([abc])\s+to\s+([abc])", re.I)


def parse_moves(text: str) -> list[tuple[int, str, str]]:
    moves = []
    for m in _MOVE_RE.finditer(text or ""):
        moves.append((int(m.group(1)), m.group(2).upper(), m.group(3).upper()))
    return moves


@dataclass
class HanoiResult:
    solved: bool
    n_moves: int
    n_invalid: int
    optimal: int
    reached_goal: bool
    parsed_any: bool


def score(initial: State, n: int, text: str, step_cap: int = 10) -> HanoiResult:
    moves = parse_moves(text)
    s = clone(initial)
    goal = state_key(goal_state(n))
    n_invalid = 0
    applied = 0
    for (num, src, trg) in moves[:step_cap]:
        if is_valid_move(s, num, src, trg):
            s = apply_move(s, num, src, trg)
            applied += 1
            if state_key(s) == goal:
                break
        else:
            n_invalid += 1
    reached = state_key(s) == goal
    # "% solved" = reached goal with ZERO invalid moves proposed (paper's strict sense)
    solved = reached and n_invalid == 0
    return HanoiResult(
        solved=solved, n_moves=applied, n_invalid=n_invalid,
        optimal=optimal_length(initial, n), reached_goal=reached,
        parsed_any=bool(moves),
    )
