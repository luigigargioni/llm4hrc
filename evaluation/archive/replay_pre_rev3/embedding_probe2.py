"""Second probe: a stronger embedder, and embeddings as a verifier.

Probe A showed all-minilm reaching recall@3 = 100% on the stress test but
only 7 of 12 on the verbose activity names real traffic produces. Two
questions follow.

1. Is that the embedder or the task? Re-run with OpenAI
   text-embedding-3-small on the same inputs.
2. The catalogue is 26 actions and already fits in the prompt, so
   retrieval cannot buy context room; it could only bias the choice. The
   job embeddings might actually fit is the opposite one: checking a
   mapping the LLM has already made, which is what the validator's
   keyword-overlap consistency check does today. Measure whether
   similarity separates correct mappings from wrong ones.
"""
import json
import os
import sys
import collections

sys.path.insert(0, r"C:\Users\Gargioni\Il mio Drive\Academic\_Unibs\PhD\Hybrid Architecture\Extension Planner\llm4hrc")
os.chdir(r"C:\Users\Gargioni\Il mio Drive\Academic\_Unibs\PhD\Hybrid Architecture\Extension Planner\llm4hrc")

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import numpy as np
from openai import OpenAI

from planning.canonical_actions import CANONICAL_ACTIONS
from evaluation.canonicalizer_stress_test import STRESS_TEST_ITEMS

client = OpenAI()
MODEL = "text-embedding-3-small"


def embed(texts):
    out = client.embeddings.create(model=MODEL, input=texts)
    vectors = np.array([d.embedding for d in out.data], dtype=float)
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)


names = list(CANONICAL_ACTIONS)
docs = [f"{n.replace('_', ' ')}: {CANONICAL_ACTIONS[n].get('description', '')}"
        for n in names]
action_vecs = embed(docs)
index = {n: i for i, n in enumerate(names)}

# ---------------------------------------------------------------- probe 1
items = STRESS_TEST_ITEMS
item_vecs = embed([i["input"] for i in items])
sims = item_vecs @ action_vecs.T
order = np.argsort(-sims, axis=1)
in_scope = [i for i, it in enumerate(items) if it["expected"] != "unknown"]
out_scope = [i for i, it in enumerate(items) if it["expected"] == "unknown"]

print(f"=== {MODEL} sullo stress test")
for k in (1, 3, 5):
    hit = sum(1 for i in in_scope
              if items[i]["expected"] in [names[j] for j in order[i][:k]])
    print(f"  recall@{k}: {hit}/{len(in_scope)} ({hit/len(in_scope):.0%})")
top_in = np.array([sims[i][order[i][0]] for i in in_scope])
top_out = np.array([sims[i][order[i][0]] for i in out_scope])
print(f"  primo risultato: in-scope media {top_in.mean():.3f} (min {top_in.min():.3f}), "
      f"out-of-scope media {top_out.mean():.3f} (max {top_out.max():.3f})")
print(f"  out-of-scope sopra il minimo in-scope: "
      f"{sum(1 for s in top_out if s >= top_in.min())}/{len(out_scope)}")

# ---------------------------------------------------------------- probe 2
seen = {}
import glob
for f in sorted(glob.glob('evaluation/archive/replay_pre_rev3/final_P*.json')) + ['evaluation/archive/replay_pre_rev3/replay_results_full.json']:
    try:
        d = json.load(open(f, encoding='utf-8'))
    except Exception:
        continue
    for t in d.get('transcripts', []):
        k = t['transcript']
        if k not in seen or t.get('turns_replayed', 0) > seen[k].get('turns_replayed', 0):
            seen[k] = t
mapped = collections.defaultdict(set)
for t in seen.values():
    for c in t.get('canonicalizations', []):
        mapped[c['activity_name']].add(c.get('canonical_action'))
unstable = {n: v for n, v in mapped.items() if len(v) > 1}

real_names = sorted(unstable)
real_vecs = embed(real_names)
rsims = real_vecs @ action_vecs.T
rorder = np.argsort(-rsims, axis=1)
print(f"\n=== {MODEL} sui nomi instabili del corpus ({len(real_names)})")
hits = 0
for i, name in enumerate(real_names):
    top3 = [names[j] for j in rorder[i][:3]]
    known = {a for a in unstable[name] if a and a != 'unknown'}
    ok = bool(known & set(top3))
    hits += ok
    print(f"  [{'si' if ok else 'NO'}] {name!r} -> {top3}  (mappato {sorted(unstable[name])})")
print(f"  azione contesa nei primi 3: {hits}/{len(real_names)}")

# ---------------------------------------------------------- probe 3: verifier
# Similarity between an activity name and the action it was mapped to,
# for mappings known to be right and known to be wrong.
print(f"\n=== embedding come verificatore (stress test, ground truth nota)")
right, wrong = [], []
for i, it in enumerate(items):
    exp = it["expected"]
    if exp == "unknown":
        continue
    right.append(sims[i][index[exp]])
    # the most plausible wrong action: the best-scoring one that is not it
    for j in order[i]:
        if names[j] != exp:
            wrong.append(sims[i][j])
            break
right, wrong = np.array(right), np.array(wrong)
print(f"  mappatura corretta:  media {right.mean():.3f}, min {right.min():.3f}")
print(f"  miglior alternativa: media {wrong.mean():.3f}, max {wrong.max():.3f}")
print(f"  casi in cui l'alternativa batte la corretta: "
      f"{int((wrong > right).sum())}/{len(right)}")
# a threshold that would flag a mapping as suspicious
for thr in (0.30, 0.35, 0.40, 0.45):
    flagged_right = int((right < thr).sum())
    print(f"  soglia {thr}: mappature corrette segnalate a torto {flagged_right}/{len(right)}")
