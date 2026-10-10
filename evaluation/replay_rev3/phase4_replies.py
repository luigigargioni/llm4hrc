"""Phase 4: the robot's replies next to the validator's decisions.

For every reply of a replayed session, prints the simulated time, the
activity the reply declared, the status the validator gave (layer arm) or
would have given (nolayer arm) to the meal-dependent medications and meals
in the same cycle, and any override. Rows where a medication was blocked
for a missing meal are the ones to read by hand.

    python evaluation/replay_rev3/phase4_replies.py <arm> <label> [...]
    python evaluation/replay_rev3/phase4_replies.py nolayer P06 --csv out.csv
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WATCH = ("ibuprofen", "vitamin", "breakfast", "lunch")


def rows_for(arm: str, label: str) -> list[dict]:
    result = json.load(open(HERE / "phase4" / arm / f"{label}.json",
                            encoding="utf8"))["transcripts"][0]
    rows, current, statuses = [], None, {}
    for line in open(result["structured_log"], encoding="utf8"):
        entry = json.loads(line)
        kind = entry.get("decision_type")
        if kind == "temporal_check":
            current = entry.get("current_time")
        name = str(entry.get("activity_name", ""))
        watched = any(w in name.lower() for w in WATCH)
        if kind == "arbitration" and arm == "layer" and watched:
            statuses[name] = (entry["arbitrated_status"],
                              entry["not_executable_reasons"])
        if kind == "validator_disabled" and watched:
            statuses[name] = (entry["would_be_status"],
                              entry["would_be_reasons"])
        if kind == "hard_enforcement":
            rows[-1]["override"] = entry.get("vocal_answer_after", "")
        if kind == "reply":
            meal_blocked = sorted(
                n for n, (status, reasons) in statuses.items()
                if any("after_meal" in r or "patient_fed" in r
                       for r in reasons)
            )
            rows.append({
                "arm": arm,
                "session": label,
                "time": current,
                "declared": entry.get("proposed_activity"),
                "blocked_for_meal": "; ".join(meal_blocked),
                "reply": entry.get("vocal_answer") or "",
                "override": "",
                "proposes_blocked": "",
            })
            statuses = {}
    return rows


def main(argv: list[str]) -> None:
    out = None
    if "--csv" in argv:
        i = argv.index("--csv")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    arm, labels = argv[0], argv[1:]
    rows = [r for label in labels for r in rows_for(arm, label)]
    for r in rows:
        flag = " <-- medication blocked for a meal" if r["blocked_for_meal"] \
            else ""
        print(f"{r['session']} {r['time']} | declared: {r['declared']} | "
              f"{r['reply'][:150]}{flag}")
        if r["override"]:
            print(f"    OVERRIDE -> {r['override'][:150]}")
    if out:
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
