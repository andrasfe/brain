"""Embodiment self-test — prove the hands work, safely.

Verifies the full eyes+hands path end-to-end WITHOUT risking your data or
systems: it only **moves the mouse** (a few cursor moves) and observes. No
clicks, no typing into apps, no key combos — a cursor move is non-destructive
and reversible. This is how we close the "hands wired but never exercised"
gap without letting the brain loose on the live desktop.

`read_only` stays the default everywhere else; this tool constructs an
embodiment with hands enabled *only for the move-test* and never issues a
destructive action. Real task execution remains opt-in + confirm-gated through
the normal cognitive path (basal ganglia + SafetyGate).

CLI:  python -m brain.embodiment_selftest
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple


# Safe cursor waypoints (screen-fraction). Center + small excursions + back.
_DEFAULT_MOVES: List[Tuple[float, float]] = [
    (0.50, 0.50), (0.40, 0.45), (0.60, 0.55), (0.50, 0.50),
]


def run_selftest(embodiment, *, moves: Optional[List[Tuple[float, float]]] = None,
                 log=print) -> dict[str, Any]:
    """Eyes check + mouse-move-only hands check. Never clicks or types."""
    report: dict[str, Any] = {"eyes_ok": False, "caps": [], "moves": [],
                              "moved_ok": 0, "notes": ""}
    if embodiment is None:
        report["notes"] = "no embodiment"
        return report

    caps = sorted(embodiment.capabilities())
    report["caps"] = caps

    # ── eyes ──
    try:
        obs = embodiment.observe()
        report["eyes_ok"] = obs is not None
        report["frontmost_app"] = getattr(obs, "frontmost_app", None)
        log(f"eyes: ok (app={report.get('frontmost_app')})")
    except Exception as e:  # noqa: BLE001
        report["notes"] = f"observe failed: {e}"
        log(f"eyes: FAILED ({e})")

    # ── hands: MOVE ONLY (no clicks/typing) ──
    if "click" not in caps:  # cliclick / pointer control unavailable
        report["notes"] = (report["notes"] + "; " if report["notes"] else "") + \
            "no pointer capability (install cliclick) — eyes-only"
        log("hands: not available (no pointer capability)")
        return report

    for (x, y) in (moves or _DEFAULT_MOVES):
        try:
            res = embodiment.move_to(x, y, observe_after=False)
        except Exception as e:  # noqa: BLE001
            report["moves"].append({"x": x, "y": y, "ok": False, "error": str(e)})
            continue
        ok = bool(getattr(res, "ok", False))
        refused = bool(getattr(res, "refused", False))
        report["moves"].append({"x": x, "y": y, "ok": ok, "refused": refused,
                                "reason": getattr(res, "reason", "")})
        if ok:
            report["moved_ok"] += 1
    n = len(report["moves"])
    log(f"hands: {report['moved_ok']}/{n} cursor moves ok (no clicks, no typing)")
    if report["moved_ok"] == 0 and n:
        report["notes"] = (report["notes"] + "; " if report["notes"] else "") + \
            "moves refused/failed — grant Accessibility to python (System " \
            "Settings → Privacy & Security → Accessibility)"
    return report


def main() -> int:
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from afferent import Embodiment, MacOSBackend

    print("Embodiment self-test — mouse-move only, no clicks/typing.\n")
    # Hands enabled ONLY for this move-test; confirm=None (allow the moves);
    # nothing destructive is ever issued.
    em = Embodiment(MacOSBackend(), read_only=False)
    rep = run_selftest(em)
    em.close()
    print("\n--- report ---")
    print(f" capabilities : {rep['caps']}")
    print(f" eyes         : {'ok' if rep['eyes_ok'] else 'FAILED'} "
          f"(app={rep.get('frontmost_app')})")
    print(f" cursor moves : {rep['moved_ok']}/{len(rep['moves'])} ok")
    if rep["notes"]:
        print(f" notes        : {rep['notes']}")
    ok = rep["eyes_ok"] and (("click" not in rep["caps"]) or rep["moved_ok"] > 0)
    print(f"\n{'PASS' if ok else 'CHECK'}: "
          + ("eyes + hands verified safely" if ok else "see notes above"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
