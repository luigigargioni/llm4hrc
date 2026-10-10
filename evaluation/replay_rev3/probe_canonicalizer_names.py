"""Map activity names with the real canonicalizer node, memory off.

Used to check that a dictionary change does not move how the care-plan
names are mapped.

    python evaluation/replay_rev3/probe_canonicalizer_names.py <reps> <name> ...
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env", override=True)
os.environ["CANONICALIZATION_CACHE"] = "false"
import graph.graph_init  # noqa: F401,E402
from nodes.action_canonicalizer_node import action_canonicalizer_node  # noqa
from copy import deepcopy  # noqa: E402
from datetime import datetime  # noqa: E402
from utils.interfaces import INITIAL_GRAPH_STATE  # noqa: E402

reps, names = int(sys.argv[1]), sys.argv[2:]
results = collections.defaultdict(collections.Counter)
for _ in range(reps):
    state = deepcopy(INITIAL_GRAPH_STATE)
    state["fake_datetime"] = datetime(2026, 9, 25, 8, 0)
    state["updated_task"] = {"activities": [
        {"activity_name": n, "done": False, "skipped": False} for n in names
    ]}
    out = action_canonicalizer_node(state)
    for a in out["updated_task"]["activities"]:
        results[a["activity_name"]][a.get("canonical_action")] += 1
for name in names:
    print(f"{name!r}: {dict(results[name])}")
