"""Feasibility probe for an embedding shortlist in the canonicalizer.

Question: if the canonicalizer were given the k nearest canonical actions
instead of the whole catalogue, would the right one be in the list, and
would an out-of-scope request be separable from an in-scope one?

Two things decide it:
  * recall@k on the 40 in-scope stress items (the right action must be
    there, or the shortlist makes things worse than the full catalogue);
  * the similarity margin on the 10 out-of-scope items, because a
    shortlist always contains something plausible, and the alias
    experiment already showed that a wider vocabulary turns safe
    "unknown" answers into wrong-known mappings.

Runs locally against Ollama (all-minilm), no tokens spent.
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\Gargioni\Il mio Drive\Academic\_Unibs\PhD\Hybrid Architecture\Extension Planner\llm4hrc")
os.chdir(r"C:\Users\Gargioni\Il mio Drive\Academic\_Unibs\PhD\Hybrid Architecture\Extension Planner\llm4hrc")

import numpy as np
from langchain_ollama import OllamaEmbeddings

from planning.canonical_actions import CANONICAL_ACTIONS
from evaluation.canonicalizer_stress_test import STRESS_TEST_ITEMS

EMB = OllamaEmbeddings(model=os.getenv("EMBEDDINGS_MODEL", "all-minilm"))


def embed(texts):
    vectors = np.array(EMB.embed_documents(texts), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


def action_documents(variant):
    """Three ways of describing an action to the embedder."""
    docs, names = [], []
    for name, d in CANONICAL_ACTIONS.items():
        label = name.replace("_", " ")
        if variant == "name":
            text = label
        elif variant == "name+desc":
            text = f"{label}: {d.get('description', '')}"
        else:
            text = (f"{label}: {d.get('description', '')}. "
                    f"Also called: {', '.join(d.get('aliases', []))}")
        docs.append(text)
        names.append(name)
    return names, docs


def report(variant):
    names, docs = action_documents(variant)
    action_vecs = embed(docs)

    items = STRESS_TEST_ITEMS
    item_vecs = embed([i["input"] for i in items])
    sims = item_vecs @ action_vecs.T          # cosine, both normalised
    order = np.argsort(-sims, axis=1)

    in_scope = [i for i, it in enumerate(items) if it["expected"] != "unknown"]
    out_scope = [i for i, it in enumerate(items) if it["expected"] == "unknown"]

    print(f"\n=== catalogo come '{variant}' ===")
    for k in (1, 3, 5, 8):
        hit = sum(1 for i in in_scope
                  if items[i]["expected"] in [names[j] for j in order[i][:k]])
        print(f"  recall@{k} sugli in-scope: {hit}/{len(in_scope)} "
              f"({hit/len(in_scope):.0%})")

    top_in = np.array([sims[i][order[i][0]] for i in in_scope])
    top_out = np.array([sims[i][order[i][0]] for i in out_scope])
    print(f"  similarità del primo risultato: in-scope media {top_in.mean():.3f} "
          f"(min {top_in.min():.3f}), out-of-scope media {top_out.mean():.3f} "
          f"(max {top_out.max():.3f})")
    overlap = sum(1 for s in top_out if s >= top_in.min())
    print(f"  out-of-scope che superano il minimo in-scope: {overlap}/{len(out_scope)}"
          "   <- quanti una soglia non può separare")

    # per-categoria, dove il recall fallisce
    misses = [(items[i]["input"], items[i]["expected"], items[i]["category"],
               [names[j] for j in order[i][:5]])
              for i in in_scope
              if items[i]["expected"] not in [names[j] for j in order[i][:5]]]
    if misses:
        print("  in-scope fuori dai primi 5:")
        for text, exp, cat, top in misses:
            print(f"    {text!r} (atteso {exp}, {cat}) -> {top}")
    return names, action_vecs


def corpus_names():
    """Activity names seen on real traffic, with how they were mapped."""
    import glob, collections
    seen = {}
    for f in sorted(glob.glob('evaluation/archive/replay_pre_rev3/final_P*.json')) + [
            'evaluation/archive/replay_pre_rev3/replay_results_full.json']:
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
    return mapped


def corpus_report(names, action_vecs):
    mapped = corpus_names()
    unstable = {n: v for n, v in mapped.items() if len(v) > 1}
    always_unknown = {n for n, v in mapped.items() if v == {'unknown'}}
    print(f"\n=== traffico reale: {len(mapped)} nomi distinti, "
          f"{len(unstable)} instabili, {len(always_unknown)} sempre unknown")

    targets = sorted(unstable) + sorted(always_unknown)
    if not targets:
        return
    vecs = embed(targets)
    sims = vecs @ action_vecs.T
    order = np.argsort(-sims, axis=1)
    print("\n  nomi instabili: l'azione contesa è nei primi 3?")
    for idx, name in enumerate(sorted(unstable)):
        top3 = [names[j] for j in order[idx][:3]]
        known = {a for a in unstable[name] if a and a != 'unknown'}
        mark = "si" if known & set(top3) else "NO"
        print(f"    [{mark}] {name!r} mappato {sorted(unstable[name])} "
              f"-> top3 {top3} (sim {sims[idx][order[idx][0]]:.3f})")
    print("\n  nomi sempre 'unknown': cosa suggerirebbe la shortlist?")
    for offset, name in enumerate(sorted(always_unknown)):
        idx = len(unstable) + offset
        top3 = [names[j] for j in order[idx][:3]]
        print(f"    {name!r} -> top3 {top3} (sim {sims[idx][order[idx][0]]:.3f})")


if __name__ == "__main__":
    best = None
    for variant in ("name", "name+desc", "name+desc+aliases"):
        best = report(variant)
    corpus_report(*best)
