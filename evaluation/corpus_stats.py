"""Recompute the replay-corpus figures of the manuscript's replay section.

Usage: corpus_stats.py <user-study dir> <final_*.json> ...
"""
import collections
import json
import re
import sys
from pathlib import Path

study_dir = Path(sys.argv[1])
files = sys.argv[2:]

NAME_RE = re.compile(r"""['"]activity_name['"]\s*:\s*['"]([^'"]+)['"]""")

# Activity names in the original user-study logs, per session folder.
study_names = {}
for log in study_dir.glob("*/*.txt"):
    if log.name.endswith("_conversation.txt"):
        continue
    sess = re.sub(r"_log$", "", log.stem)
    text = log.read_text(encoding="cp1252", errors="replace")
    study_names[sess] = collections.Counter(
        n.strip().lower() for n in NAME_RE.findall(text))

results = []
for f in files:
    for r in json.load(open(f, encoding="utf-8"))["transcripts"]:
        results.append(r)

all_study = set().union(*[set(c) for c in study_names.values()])
print("sessions", len(results))
print("distinct names in user-study logs:", sorted(all_study))

pairs = 0
unstable = []
never_known = 0
study_pairs = study_known = 0
introduced = 0
unmapped_from_study = 0
unstable_names = collections.Counter()
switch = 0
kinds = collections.Counter()
for r in results:
    sess = r["transcript"].split("/")[1]
    seen = collections.defaultdict(list)
    for c in r["canonicalizations"]:
        seen[c["activity_name"]].append(c["canonical_action"])
    statuses = collections.defaultdict(set)
    for v in r["validations"]:
        s = v["arbitrated_status"] or "NONE"
        statuses[v["activity_name"]].add(
            "EXEC" if s == "EXECUTABLE" else
            "BLOCK" if s.startswith("BLOCKED") else s)
    names_here = study_names.get(sess, collections.Counter())
    for name, acts in seen.items():
        pairs += 1
        known = {a for a in acts if a and a != "unknown"}
        in_study = name.strip().lower() in all_study
        if in_study:
            study_pairs += 1
            study_known += bool(known)
        else:
            introduced += 1
        if not known:
            never_known += 1
            unmapped_from_study += in_study
        if len(set(acts)) > 1:
            unstable.append((r["transcript"], name))
            unstable_names[name] += 1
            if {"EXEC", "BLOCK"} <= statuses[name]:
                switch += 1
            has_unknown = "unknown" in acts
            if len(known) == 1 and has_unknown:
                kinds["known<->unknown"] += 1
            elif len(known) > 1 and has_unknown:
                kinds["several known + unknown"] += 1
            else:
                kinds["two known, no unknown"] += 1

print("name-session pairs:", pairs)
print("unstable pairs:", len(unstable), f"({len(unstable)/pairs:.0%})")
print("  status switched blocked<->executable:", switch)
print("  kinds:", dict(kinds))
print("distinct unstable names:", len(unstable_names),
      "in >=3 sessions:", sum(1 for n in unstable_names.values() if n >= 3))
print("  ", dict(unstable_names.most_common()))
print("pairs from care-plan names:", study_pairs, "mapped known:", study_known)
print("introduced pairs:", introduced)
print("never-known pairs:", never_known, "of which care-plan:",
      unmapped_from_study)

# Cost figures
tok_in = collections.Counter()
for r in results:
    for node, u in r["tokens_by_node"].items():
        tok_in[node] += u["input"]
total_in = sum(tok_in.values())
print("input share:", {k: f"{v/total_in:.1%}" for k, v in tok_in.items()})
cs = sum(r["node_seconds"].get("action_canonicalizer", 0) for r in results)
vs = sum(r["node_seconds"].get("plan_validator", 0) for r in results)
nd = sum(len(r["validations"]) for r in results)
print(f"canon {cs:.1f}s validator {vs:.2f}s ratio {cs/vs:.0f} "
      f"decisions {nd} ms/decision {1000*vs/nd:.2f}")
print("turns", sum(r["turns_replayed"] for r in results),
      "canonicalizations", sum(r["canonicalization_count"] for r in results))
