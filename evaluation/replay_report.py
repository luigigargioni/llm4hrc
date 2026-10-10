"""Consolidate replay runs into the figures the paper needs.

The corpus was replayed across several invocations (the first full pass, a
recovery pass for sessions lost to an Ollama outage and to exhausted API
credits, plus two single-session runs). This merges those result files,
keeping the most complete run of each session, and reports:

  * per-session outcome and how much of the transcript was actually read,
  * Action Canonicalizer cost against Plan Validator cost,
  * canonicalizer instability: the same activity name mapped to different
    canonical actions within one session,
  * hard-enforcement overrides, split into the ones that replaced a refusal
    (false positives of the substring check) and the rest,
  * the distribution of arbitrated statuses,
  * Goal Planner decomposition success.

Usage::

    python -m evaluation.replay_report evaluation/archive/replay_pre_rev3/replay_results_*.json
"""

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Outcomes ranked by how much evidence they carry; when one session appears
# in several runs we keep the best.
OUTCOME_RANK = {
    "transcript_exhausted": 4,
    "stalled": 3,
    "token_budget_exceeded": 3,
    "budget_exceeded": 3,
    "error": 1,
}

# A refusal names the blocked activity, so the substring check fires even
# though nothing was proposed. These markers identify such an answer.
REFUSAL_MARKERS = (
    "not available", "cannot", "can't", "can not", "expired",
    "not yet", "no longer", "unable", "won't be able", "not possible",
)


def load_runs(patterns: list[str]) -> dict[str, dict]:
    """Return the best available result per session across all result files."""
    best: dict[str, dict] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            for result in payload.get("transcripts", []):
                result = {**result, "_source": Path(path).name}
                key = result["transcript"]
                incumbent = best.get(key)
                if incumbent is None or _score(result) > _score(incumbent):
                    best[key] = result
    return best


def _score(result: dict) -> tuple[int, int]:
    return (
        OUTCOME_RANK.get(result["outcome"], 0),
        result.get("turns_replayed", 0),
    )


def declaration_compliance(results: list[dict]) -> dict:
    """How often replies declared the activity they were proposing.

    The declared-intent rule only decides when the field is there; every
    reply without it falls back to name matching, so this is the number
    that says how much of the corpus the new rule actually governs.
    Empty for runs logged before the field existed.
    """
    stats = {"replies": 0, "declaration_present": 0, "declared_null": 0,
             "declared_activity": 0}
    for result in results:
        log_path = result.get("structured_log")
        if not log_path or not os.path.exists(log_path):
            continue
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("decision_type") != "reply":
                    continue
                stats["replies"] += 1
                if entry.get("declaration_present"):
                    stats["declaration_present"] += 1
                    if entry.get("proposed_activity"):
                        stats["declared_activity"] += 1
                    else:
                        stats["declared_null"] += 1
    return stats


def classify_overrides(results: list[dict]) -> list[dict]:
    """Pair each override with the answer it replaced, read from its log."""
    classified = []
    for result in results:
        log_path = result.get("structured_log")
        if not log_path or not os.path.exists(log_path):
            continue
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("decision_type") != "hard_enforcement":
                    continue
                before = (entry.get("vocal_answer_before") or "")
                lowered = before.lower()
                is_refusal = any(m in lowered for m in REFUSAL_MARKERS)
                classified.append({
                    "session": result["transcript"],
                    "replaced_a_refusal": is_refusal,
                    "robot_action_before": entry.get("robot_action_before"),
                    "robot_action_after": entry.get("robot_action_after"),
                    "vocal_answer_before": before,
                    # Absent in logs written before declared-intent
                    # enforcement; present runs split by which rule fired.
                    "match_mode": entry.get("match_mode"),
                    "declared_activity": entry.get("declared_activity"),
                })
    return classified


def unstable_mappings(results: list[dict]) -> dict[str, dict[str, dict]]:
    """Activity names that received more than one canonical action."""
    unstable: dict[str, dict[str, dict]] = {}
    for result in results:
        seen: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for record in result["canonicalizations"]:
            seen[record["activity_name"]][record["canonical_action"]] += 1
        multi = {
            name: dict(counts)
            for name, counts in seen.items()
            if len(counts) > 1
        }
        if multi:
            unstable[result["transcript"]] = multi
    return unstable


def goal_planner_stats(results: list[dict]) -> dict:
    """How often backward chaining produced a chain instead of falling back."""
    attempts = successes = fallbacks = 0
    for result in results:
        log_path = result.get("structured_log")
        if not log_path or not os.path.exists(log_path):
            continue
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("decision_type") != "goal_decomposition":
                    continue
                attempts += 1
                if entry.get("decomposition_success"):
                    successes += 1
                if entry.get("fallback_used"):
                    fallbacks += 1
    return {
        "attempts": attempts,
        "decomposed": successes,
        "fell_back_to_llm": fallbacks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate replay runs")
    parser.add_argument("results", nargs="*",
                        default=["evaluation/archive/replay_pre_rev3/replay_results*.json"],
                        help="result files or globs to merge")
    parser.add_argument("--out", default="evaluation/replay_report.json")
    args = parser.parse_args()

    best = load_runs(args.results)
    results = sorted(best.values(), key=lambda r: r["transcript"])
    if not results:
        parser.error("nessun file di risultati trovato")

    overrides = classify_overrides(results)
    overrides_by_session = collections.Counter(o["session"] for o in overrides)

    # 'cand' counts replies whose action has the shape an override leaves
    # behind, which the model also chooses on its own; 'over' is the
    # authoritative count, from the hard_enforcement entries in the log.
    print(f"=== SESSIONI ({len(results)}) ===")
    header = (f"{'sessione':26} {'turni':>5} {'letti':>6} {'cicli':>6} "
              f"{'cand':>5} {'over':>5}  esito")
    print(header)
    read_all = 0
    for r in results:
        cycles = r["node_calls"].get("action_canonicalizer", 0)
        complete = r["turns_replayed"] >= r["turns_in_transcript"]
        read_all += complete
        print(f"{r['transcript']:26} {r['turns_in_transcript']:5} "
              f"{r['turns_replayed']:6} {cycles:6} "
              f"{len(r['hard_enforcements']):5} "
              f"{overrides_by_session[r['transcript']]:5}  {r['outcome']}")
    print(f"\nsessioni lette per intero: {read_all}/{len(results)}")

    canon_s = sum(r["node_seconds"].get("action_canonicalizer", 0.0)
                  for r in results)
    valid_s = sum(r["node_seconds"].get("plan_validator", 0.0)
                  for r in results)
    canon_calls = sum(r["canonicalization_count"] for r in results)
    print("\n=== COSTO CANONICALIZER vs VALIDATOR ===")
    print(f"action_canonicalizer : {canon_s:9.2f} s  ({canon_calls} chiamate)")
    print(f"plan_validator       : {valid_s:9.2f} s")
    if valid_s > 0:
        print(f"rapporto             : {canon_s / valid_s:9.1f} : 1")

    tokens: dict[str, dict[str, int]] = {}
    for r in results:
        for node, usage in r.get("tokens_by_node", {}).items():
            bucket = tokens.setdefault(node, {"calls": 0, "input": 0,
                                              "output": 0})
            for field in ("calls", "input", "output"):
                bucket[field] += usage[field]
    print("\n=== TOKEN PER NODO ===")
    for node, usage in sorted(tokens.items(),
                              key=lambda kv: -kv[1]["input"]):
        print(f"{node:22} {usage['calls']:5} chiamate  "
              f"{usage['input']:9} in  {usage['output']:7} out")

    statuses: collections.Counter = collections.Counter()
    precondition_blocks = 0
    for r in results:
        for v in r["validations"]:
            statuses[v["arbitrated_status"] or "NONE"] += 1
            if any("missing_precondition" in reason
                   for reason in v["reasons"]):
                precondition_blocks += 1
    print("\n=== STATI ARBITRATI ===")
    total_decisions = sum(statuses.values())
    for status, n in statuses.most_common():
        print(f"{status:24} {n:6}  ({n / total_decisions:5.1%})")
    print(f"{'di cui da precondizione':24} {precondition_blocks:6}  "
          f"({precondition_blocks / total_decisions:5.1%})")

    unstable = unstable_mappings(results)
    print("\n=== MAPPATURE INSTABILI ===")
    affected = sum(len(v) for v in unstable.values())
    print(f"attività con piu' di una azione canonica: {affected} "
          f"in {len(unstable)} sessioni")
    for session, mappings in unstable.items():
        for name, counts in mappings.items():
            spread = " | ".join(f"{a} ({n})" for a, n in
                                sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f"  {session}  {name!r} -> {spread}")

    print("\n=== HARD ENFORCEMENT ===")
    if overrides:
        fp = sum(1 for o in overrides if o["replaced_a_refusal"])
        print(f"override totali: {len(overrides)}")
        print(f"  hanno sostituito un rifiuto gia' corretto: {fp} "
              f"({fp / len(overrides):.0%})  <- falsi positivi")
        print(f"  altri (da verificare a mano): {len(overrides) - fp}")
        modes = collections.Counter(
            o.get("match_mode") or "unrecorded" for o in overrides
        )
        print("  regola che ha deciso: " + ", ".join(
            f"{mode} {n}" for mode, n in modes.most_common()
        ))
    else:
        print("nessun override registrato con il testo pre-override")

    declarations = declaration_compliance(results)
    if declarations["replies"]:
        print(f"risposte registrate: {declarations['replies']}, "
              f"con dichiarazione: {declarations['declaration_present']} "
              f"({declarations['declaration_present'] / declarations['replies']:.0%})"
              f" - di cui null {declarations['declared_null']}, "
              f"con attivita' {declarations['declared_activity']}")

    print("\n=== GOAL PLANNER ===")
    print(goal_planner_stats(results))

    payload = {
        "sessions": results,
        "canonicalizer_seconds": round(canon_s, 2),
        "validator_seconds": round(valid_s, 2),
        "tokens_by_node": tokens,
        "arbitrated_status_counts": dict(statuses),
        "precondition_blocks": precondition_blocks,
        "unstable_mappings": unstable,
        "hard_enforcement": overrides,
        "declaration_compliance": declarations,
        "goal_planner": goal_planner_stats(results),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nreport: {args.out}")


if __name__ == "__main__":
    main()
