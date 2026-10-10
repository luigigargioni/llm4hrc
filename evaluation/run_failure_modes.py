"""Entry point for the failure mode robustness analysis (Fase B.4).

Usage:
    python evaluation/run_failure_modes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.failure_mode_runner import run_failure_modes


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "failure_mode_results.json",
    )

    print("=" * 60)
    print("  FAILURE MODE ROBUSTNESS ANALYSIS (Fase B.4)")
    print("  45 fault injection scenarios")
    print("=" * 60)
    print()

    results = run_failure_modes(output_file)

    summary = results["summary"]
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total fault injections:  {summary['total_scenarios']}")
    print(f"  Detected:                {summary['detected']}")
    print(f"  Undetected:              {summary['undetected']}")
    print(f"  Detection rate:          {summary['detection_rate']:.1%}")
    print()
    print("  Per fault type:")
    for ft, stats in summary["fault_type_breakdown"].items():
        print(f"    {ft:30s} {stats['detected']}/{stats['total']} "
              f"({stats['detection_rate']:.0%}) "
              f"[test_pass: {stats['test_pass']}/{stats['total']}]")
    print()
    print(f"  Results saved to: {output_file}")
    print("=" * 60)

    # Print undetected faults for analysis
    undetected = [r for r in results["scenarios"] if not r["actual_detection"]]
    if undetected:
        print(f"\n  UNDETECTED FAULTS ({len(undetected)}):")
        for r in undetected:
            print(f"    - {r['id']} ({r['fault_type']}): {r['description']}")
            print(f"        Error: {r['injected_error']}")
            print(f"        Reason: {r['expected_reason']}")
