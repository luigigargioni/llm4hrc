"""Quick checks on one block of the revision-3 enforcement benchmark.

    python evaluation/enforcement_rev3/check_block.py <results.json>
"""
import json
import re
import sys

TECH = re.compile(r"BLOCKED_CRITICAL|BLOCKED_NON_CRITICAL|\bDEFERRED\b"
                  r"|\bEXECUTABLE\b|\b[a-z]+_[a-z_]+\b")
PRECOND = re.compile(r"pre-?\u2011?condition", re.I)
RECORD = re.compile(r"\bnote that you|mark (you|that|the)|record (that )?you",
                    re.I)

replies = json.load(open(sys.argv[1], encoding="utf8"))["replies"]
print(len(replies), "replies")
for arm in ("declared", "legacy"):
    arm_replies = [r for r in replies if r["arm"] == arm]
    print(arm, len(arm_replies),
          "| declaration", sum(bool(r.get("declaration_present"))
                               for r in arm_replies),
          "| 'precondition'", sum(bool(PRECOND.search(r["vocal_answer"]))
                                  for r in arm_replies),
          "| identifiers", sum(bool(TECH.search(r["vocal_answer"]))
                               for r in arm_replies),
          "| record offers", sum(bool(RECORD.search(r["vocal_answer"]))
                                 for r in arm_replies),
          "| overrides", sum(bool(r.get("applied_override"))
                             for r in arm_replies),
          "| heuristic proposals", sum(bool(r.get("auto_proposes_blocked"))
                                       for r in arm_replies))
for r in replies:
    if TECH.search(r["vocal_answer"]) or RECORD.search(r["vocal_answer"]):
        print("  check:", r["case_id"], r["arm"], r["repeat"],
              TECH.findall(r["vocal_answer"])[:3], r["vocal_answer"][:160])
