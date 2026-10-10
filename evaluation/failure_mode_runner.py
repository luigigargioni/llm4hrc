"""Failure mode runner for robustness analysis (Fase B.4).

Runs each failure mode scenario through the symbolic pipeline and
measures whether the injected error is detected (activity blocked)
or goes undetected (activity executes despite the error).

Key distinction from the main evaluation:
  - "detected" = the plan_validator blocks the activity (even if for
    the wrong reason) OR the world_state_updater auto-corrects the state
  - "undetected" = the activity executes despite the injected error

Output: failure_mode_results.json with per-fault-type breakdown.
"""

import json
import os
import sys
import glob
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.structured_logs as structured_logs
from evaluation.failure_mode_scenarios import FAILURE_MODE_SCENARIOS
from evaluation.scenario_runner import _build_state, _run_symbolic, _extract_observed, _compare


def _read_latest_structured_logs() -> list:
    """Read all entries from the latest structured log file."""
    files = glob.glob("log/structured/*_structured.jsonl")
    if not files:
        return []
    latest = max(files, key=os.path.getmtime)
    entries = []
    try:
        with open(latest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return entries


def _count_new_entries(entries_before: list, entries_after: list, decision_type: str) -> int:
    """Count how many new entries of a given decision_type appeared."""
    before_count = sum(1 for e in entries_before if e.get("decision_type") == decision_type)
    after_count = sum(1 for e in entries_after if e.get("decision_type") == decision_type)
    return after_count - before_count


def run_failure_modes(output_file: str = "evaluation/failure_mode_results.json") -> dict:
    """Run all failure mode scenarios and produce results."""
    results = []
    detected_count = 0
    undetected_count = 0

    # Per fault type breakdown
    fault_type_stats = {}

    for scenario in FAILURE_MODE_SCENARIOS:
        sid = scenario["id"]
        desc = scenario["description"]
        fault_type = scenario["fault_type"]
        expected = scenario["expected"]
        expected_detection = scenario["expected_detection"]

        # Snapshot the structured log and its size before running, so only
        # the entries this scenario appends are counted afterwards. The
        # file is asked of the logging module rather than guessed from the
        # directory: picking the "most recent" file by mtime is ambiguous
        # when two runs fall in the same second, and the whole file was
        # then re-read, counting earlier scenarios' entries as new and
        # making the detection rate vary between identical runs.
        log_before = structured_logs._structured_log_filename  # noqa: SLF001
        log_offset = (
            os.path.getsize(log_before)
            if log_before and os.path.exists(log_before)
            else 0
        )

        state = _build_state(scenario)
        state = _run_symbolic(state)

        log_after = structured_logs._structured_log_filename  # noqa: SLF001
        new_log_entries = []
        if log_after and os.path.exists(log_after):
            same_file = log_after == log_before
            with open(log_after, "r", encoding="utf-8") as f:
                f.seek(log_offset if same_file else 0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            new_log_entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        observed = _extract_observed(state)

        # Compare observed vs. expected
        comparison = _compare(observed, expected)
        test_pass = comparison["all_pass"]

        # Determine if the error was detected
        # "Detected" = at least one activity was blocked OR
        # an invariant violation was logged OR
        # a suspicious mapping was flagged OR
        # world_state_updater auto-corrected stale state
        any_blocked = (
            len(observed["blocked_critical"]) > 0
            or len(observed["blocked_non_critical"]) > 0
            or len(observed["deferred"]) > 0
        )
        # Count new entries from this scenario's run only
        invariant_violations_logged = sum(
            1 for e in new_log_entries if e.get("decision_type") == "invariant_violation"
        )
        suspicious_mappings_logged = sum(
            1 for e in new_log_entries if e.get("decision_type") == "suspicious_mapping"
        )
        mutual_exclusivity_corrections_logged = sum(
            1 for e in new_log_entries if e.get("decision_type") == "mutual_exclusivity_correction"
        )
        idempotent_reapplications_logged = sum(
            1 for e in new_log_entries if e.get("decision_type") == "idempotent_reapplication"
        )
        malformed_temporal_inputs_logged = sum(
            1 for e in new_log_entries if e.get("decision_type") == "malformed_temporal_input"
        )

        if fault_type == "stale_world_state":
            detected = test_pass
        elif fault_type == "corrupted_world_state":
            detected = any_blocked or invariant_violations_logged > 0
        elif fault_type == "wrong_canonicalization":
            detected = any_blocked or suspicious_mappings_logged > 0
        elif fault_type == "conflicting_effects":
            # Detected if blocked, invariant violation, OR mutual-exclusivity
            # correction logged (F10-style auto-correction is now auditable)
            detected = (
                any_blocked
                or invariant_violations_logged > 0
                or mutual_exclusivity_corrections_logged > 0
                or idempotent_reapplications_logged > 0
            )
        elif fault_type == "wrong_temporal_window":
            # Detected if blocked (invalid/malformed/contradictory time) or
            # a malformed_temporal_input entry was logged
            detected = any_blocked or malformed_temporal_inputs_logged > 0
        elif fault_type == "orphan_precondition":
            # Detected if blocked with missing precondition (no recovery possible)
            detected = any_blocked
        else:
            detected = any_blocked

        if detected:
            detected_count += 1
        else:
            undetected_count += 1

        # Track per fault type
        if fault_type not in fault_type_stats:
            fault_type_stats[fault_type] = {
                "total": 0,
                "detected": 0,
                "undetected": 0,
                "test_pass": 0,
            }
        fault_type_stats[fault_type]["total"] += 1
        if detected:
            fault_type_stats[fault_type]["detected"] += 1
        else:
            fault_type_stats[fault_type]["undetected"] += 1
        if test_pass:
            fault_type_stats[fault_type]["test_pass"] += 1

        results.append({
            "id": sid,
            "description": desc,
            "fault_type": fault_type,
            "injected_error": scenario["injected_error"],
            "expected_detection": expected_detection,
            "actual_detection": detected,
            "detection_correct": detected == expected_detection,
            "detection_details": {
                "any_blocked": any_blocked,
                "invariant_violations_logged": invariant_violations_logged,
                "suspicious_mappings_logged": suspicious_mappings_logged,
                "mutual_exclusivity_corrections_logged": mutual_exclusivity_corrections_logged,
                "idempotent_reapplications_logged": idempotent_reapplications_logged,
                "malformed_temporal_inputs_logged": malformed_temporal_inputs_logged,
            },
            "test_pass": test_pass,
            "expected": expected,
            "observed": observed,
            "comparison": comparison["checks"],
            "expected_reason": scenario.get("expected_reason", ""),
        })

        status = "DETECTED" if detected else "UNDETECTED"
        pass_str = "PASS" if test_pass else "FAIL"
        print(f"  [{status}] [{pass_str}] {sid}: {desc}")

    total = len(FAILURE_MODE_SCENARIOS)
    detection_rate = detected_count / total if total > 0 else 0

    # Per fault type detection rates
    for ft, stats in fault_type_stats.items():
        stats["detection_rate"] = round(stats["detected"] / stats["total"], 4) if stats["total"] > 0 else 0

    summary = {
        "total_scenarios": total,
        "detected": detected_count,
        "undetected": undetected_count,
        "detection_rate": round(detection_rate, 4),
        "fault_type_breakdown": fault_type_stats,
    }

    output = {
        "summary": summary,
        "scenarios": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return output
