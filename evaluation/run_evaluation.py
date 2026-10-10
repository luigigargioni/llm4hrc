"""Entry point for the symbolic layer evaluation suite.

Usage:
    python evaluation/run_evaluation.py
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.scenario_runner import run_evaluation


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "evaluation_results.json",
    )

    print("=" * 60)
    print("  SYMBOLIC LAYER EVALUATION SUITE")
    print("  20 scenarios + ablation comparison")
    print("=" * 60)
    print()

    results = run_evaluation(output_file)

    summary = results["summary"]
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Scenarios:     {summary['total_scenarios']}")
    print(f"  Passed:        {summary['passed']}")
    print(f"  Failed:        {summary['failed']}")
    print(f"  Success rate:  {summary['success_rate']:.1%}")
    print()
    print(f"  Violations intercepted (symbolic):  {summary['violations_intercepted_symbolic']}")
    print(f"  Violations missed (ablation):       {summary['violations_missed_ablation']}")
    print(f"  Violations missed (Event Planner):  {summary['violations_missed_event_planner_only']}")
    print(f"  Violations missed (symbolic):       {summary['violations_missed_symbolic']}")
    print(f"  Recovery suggestions:               {summary['recovery_suggestions']}")
    print(f"  Ablation error reduction:           {summary['ablation_error_reduction']:.1%}")
    print()
    print(f"  Results saved to: {output_file}")
    print("=" * 60)

    if summary["failed"] > 0:
        print("\n  FAILED SCENARIOS:")
        for r in results["scenarios"]:
            if not r["pass"]:
                print(f"    - {r['id']}: {r['description']}")
                for check_name, check in r["comparison"].items():
                    if not check["match"]:
                        print(f"        {check_name}:")
                        print(f"          expected: {check.get('expected', [])}")
                        print(f"          observed: {check.get('observed', [])}")
                        if check.get("missing"):
                            print(f"          missing:  {check['missing']}")
                        if check.get("extra"):
                            print(f"          extra:    {check['extra']}")
        sys.exit(1)
