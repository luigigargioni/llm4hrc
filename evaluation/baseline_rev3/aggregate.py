"""Aggregate the revision-3 verifier baseline: 5 runs per model, structured
output (JSON schema), temperature 0, on the revised ground truth.

Same scoring as evaluation/baseline_reps/aggregate.py, plus the latency
median pooled over all runs (the figure the paper reports) and the number
of providers that served each OpenRouter model.

    python evaluation/baseline_rev3/aggregate.py
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
# A "-free" tag is the same model without the JSON schema, to measure what
# constrained decoding costs a model that reasons before answering.
TAGS = ["gptoss120b", "gptoss120b-free", "gpt54mini", "gpt54mini-free",
        "qwen3-30b", "qwen3-30b-free", "mistral-small", "mistral-small-free",
        "llama33-70b", "llama33-70b-free"]
BUCKETS = ["executable", "blocked_critical", "blocked_non_critical", "deferred"]


def score(d):
    sc = dr = adm = admc = perr = 0
    lat = []
    for s in d["scenarios"]:
        scripted = int(s["id"][1:]) <= 20
        exp, obs = s["expected"], s["observed"]
        lat.append(s["latency_seconds"])
        perr += s["parse_error"]
        names = [n for b in BUCKETS for n in exp.get(b, [])]
        for n in names:
            eb = next(b for b in BUCKETS if n in exp.get(b, []))
            ob = next((b for b in BUCKETS if n in obs.get(b, [])), None)
            if eb == ob:
                sc += scripted
                dr += not scripted
            if eb != "executable" and ob == "executable":
                adm += 1
                admc += eb == "blocked_critical"
    return dict(scripted=sc, derived=dr, total=sc + dr, admitted=adm,
                admitted_crit=admc, parse_errors=perr,
                mean_lat=st.mean(lat), median_lat=st.median(lat)), lat


out = {}
for tag in TAGS:
    files = sorted(glob.glob(os.path.join(HERE, f"{tag}_run[0-9].json")))
    if not files:
        continue
    data = [json.load(open(f, encoding="utf8")) for f in files]
    scored = [score(d) for d in data]
    runs = [r for r, _ in scored]
    pooled = [x for _, lat in scored for x in lat]
    providers = {s.get("provider_name") for d in data for s in d["scenarios"]
                 if s.get("provider_name")}
    agg = {k: (st.mean(r[k] for r in runs), min(r[k] for r in runs),
               max(r[k] for r in runs)) for k in runs[0]}
    out[tag] = {"runs": runs, "pooled_median_latency": st.median(pooled),
                "providers": sorted(providers)}
    print(f"{tag:14s} n={len(runs)} " + "  ".join(
        f"{k}={m:.1f}[{lo:.0f}-{hi:.0f}]" if "lat" not in k
        else f"{k}={m:.2f}" for k, (m, lo, hi) in agg.items())
        + f"  pooled_median={st.median(pooled):.2f}s"
        + (f"  providers={len(providers)}" if providers else ""))
json.dump(out, open(os.path.join(HERE, "aggregate.json"), "w"), indent=1)
