"""Recompute the hand-labeled counts reported in the paper from the label files.

Usage: python evaluation/label_summary.py

See evaluation/LABELS.md for the labeling criteria.
"""
import collections
import csv
from pathlib import Path

HERE = Path(__file__).parent

ENFORCEMENT = HERE / "enforcement_rev3" / "enforcement_labels.csv"
MEAL_REPLAY = HERE / "replay_rev3" / "analysis" / "phase4_replies_to_review.csv"
ORIGINAL_VS_LAYER = HERE / "replay_rev3" / "analysis" / "original_vs_layer_to_review.csv"

MEAL_CATEGORIES = [
    ("proposed_now", "Ibuprofen proposed before breakfast"),
    ("scheduled_without_meal", "Vitamin D scheduled without lunch"),
    ("false_statement", "False statement about the day"),
    ("out_of_dictionary_accepted", "Out-of-dictionary activity accepted"),
    ("refusal_vs_transcript", "Refusal contradicted by the transcript"),
    ("generator_over_refusal", "Refusal of an executable activity"),
    ("unclear", "Unclear (not counted in the table)"),
]


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def enforcement():
    rows = read(ENFORCEMENT)
    print(f"Enforcement benchmark (Table 8): {len(rows)} hand-labeled replies")
    counts = collections.Counter((r["arm"], r["proposes_blocked"]) for r in rows)
    for arm, name in [("declared", "declaration prompt"), ("legacy", "original prompt")]:
        print(f"  {name}: {counts[(arm, '1')]} proposals of a blocked activity, "
              f"{counts[(arm, '0')]} other replies")


def meal_replay():
    rows = read(MEAL_REPLAY)
    print(f"\nReplay with meal dependencies (Table 7): {len(rows)} replies")
    counts = collections.Counter((r["arm"], r["suggested"]) for r in rows)
    print(f"  {'':42} {'no validator':>12} {'layer':>6}")
    for key, label in MEAL_CATEGORIES:
        print(f"  {label:42} {counts[('nolayer', key)]:>12} {counts[('layer', key)]:>6}")


def original_vs_layer():
    rows = read(ORIGINAL_VS_LAYER)
    counts = collections.Counter(r["original_robot_proposes"] for r in rows)
    print(f"\nOriginal replies naming a blocked activity (Section 5.6): {len(rows)}")
    print(f"  propose the blocked activity: {counts['1']}, "
          f"conditional offer: {counts['borderline']}, do not propose: {counts['0']}")


if __name__ == "__main__":
    enforcement()
    meal_replay()
    original_vs_layer()
