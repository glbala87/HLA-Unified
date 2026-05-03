#!/usr/bin/env python3
"""Accuracy stability tracker for CI.

Appends the current validation accuracy to accuracy_history.json after
each CI run. Detects regressions by comparing the last N entries and
fails if accuracy is trending downward.

Usage:
    # After validation runs, append current results:
    python track_accuracy.py --append

    # Check stability (fails if last 3 runs show >2% decline):
    python track_accuracy.py --check

    # Both (typical CI usage):
    python track_accuracy.py --append --check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "accuracy_history.json"
REPORTS_DIR = Path(__file__).parent / "reports"
SYNTH_RESULTS = Path(__file__).parent / "real_data" / "synth_results"


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def load_history() -> dict:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return {"description": "Accuracy history", "entries": []}


def save_history(data: dict) -> None:
    HISTORY_FILE.write_text(json.dumps(data, indent=2) + "\n")


def get_current_accuracy() -> dict | None:
    """Read current accuracy from validation reports."""
    summary_path = REPORTS_DIR / "validation_summary.json"
    if not summary_path.exists():
        print(f"  No validation summary found at {summary_path}")
        return None

    summary = json.loads(summary_path.read_text())
    bench = summary.get("full_benchmark", {})

    entry = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": get_git_commit(),
        "overall_accuracy": bench.get("overall_accuracy", 0),
        "class_i_accuracy": bench.get("class_i_benchmark", {}).get("overall_accuracy",
                            summary.get("class_i_benchmark", {}).get("overall_accuracy", 0)),
        "per_locus": {
            locus: data.get("accuracy", 0)
            for locus, data in bench.get("per_locus", {}).items()
        },
        "n_samples": bench.get("n_samples", 0),
    }

    # Add synthetic BAM results if available
    synth_path = SYNTH_RESULTS / "synthetic_bam_validation.json"
    if synth_path.exists():
        synth = json.loads(synth_path.read_text())
        entry["synthetic_bam_accuracy"] = synth.get("overall_accuracy", 0)
        entry["synthetic_bam_n_samples"] = synth.get("n_samples", 0)

    return entry


def append_entry() -> bool:
    """Append current accuracy to history."""
    entry = get_current_accuracy()
    if not entry:
        print("  Cannot append — no current results available")
        return False

    history = load_history()
    history["entries"].append(entry)
    save_history(history)

    print(f"  Appended entry: accuracy={entry['overall_accuracy']:.1%}, "
          f"commit={entry['commit']}, date={entry['date']}")
    return True


def check_stability(window: int = 3, max_decline: float = 0.02) -> bool:
    """Check that accuracy hasn't declined over the last N entries.

    Args:
        window: Number of recent entries to consider
        max_decline: Maximum allowed decline from best to worst in window

    Returns:
        True if stable, False if regression detected
    """
    history = load_history()
    entries = history.get("entries", [])

    if len(entries) < 2:
        print(f"  Only {len(entries)} entries — too few to check stability")
        return True

    recent = entries[-window:]
    accuracies = [e["overall_accuracy"] for e in recent]

    best = max(accuracies)
    worst = min(accuracies)
    latest = accuracies[-1]
    decline = best - worst

    print(f"\n  Stability Check (last {len(recent)} runs):")
    for e in recent:
        synth = e.get("synthetic_bam_accuracy", "—")
        synth_str = f"{synth:.1%}" if isinstance(synth, float) else synth
        print(f"    {e['date']}  {e['commit']:<10}  "
              f"benchmark={e['overall_accuracy']:.1%}  synth_bam={synth_str}")

    print(f"\n  Best: {best:.1%}  Worst: {worst:.1%}  Latest: {latest:.1%}")
    print(f"  Max decline in window: {decline:.1%} (threshold: {max_decline:.0%})")

    if decline > max_decline:
        print(f"\n  UNSTABLE: Accuracy declined {decline:.1%} > {max_decline:.0%} threshold")
        return False

    # Also check if latest is significantly below best ever
    all_accuracies = [e["overall_accuracy"] for e in entries]
    all_time_best = max(all_accuracies)
    if latest < all_time_best - max_decline:
        print(f"\n  REGRESSION: Latest {latest:.1%} is {all_time_best - latest:.1%} "
              f"below all-time best {all_time_best:.1%}")
        return False

    print(f"\n  STABLE: Accuracy within {max_decline:.0%} tolerance")
    return True


def main():
    parser = argparse.ArgumentParser(description="Accuracy stability tracker")
    parser.add_argument("--append", action="store_true", help="Append current results")
    parser.add_argument("--check", action="store_true", help="Check stability")
    parser.add_argument("--window", type=int, default=3, help="Stability window size")
    parser.add_argument("--max-decline", type=float, default=0.02, help="Max allowed decline")
    args = parser.parse_args()

    if not args.append and not args.check:
        parser.print_help()
        return 0

    print("=" * 60)
    print("  HLA-Unified V2 — Accuracy Stability Tracker")
    print("=" * 60)

    ok = True

    if args.append:
        if not append_entry():
            ok = False

    if args.check:
        if not check_stability(args.window, args.max_decline):
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
