"""Aggregate the 5 runs per model of the LLM-verifier baseline (run 1 = original files)."""
import json, glob, statistics as st, os
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
RUN1 = {'gptoss120b': 'llm_verifier_baseline_gptoss120b.json', 'gpt54mini': 'llm_verifier_baseline_gpt54mini_28.json',
        'qwen3-30b': 'llm_verifier_baseline_qwen3-30b.json', 'mistral-small': 'llm_verifier_baseline_mistral-small.json',
        'llama33-70b': 'llm_verifier_baseline_llama33-70b.json'}
BUCKETS = ['executable', 'blocked_critical', 'blocked_non_critical', 'deferred']
def score(d):
    sc = dr = adm = admc = perr = 0; lat = []
    for s in d['scenarios']:
        scripted = int(s['id'][1:]) <= 20
        exp, obs = s['expected'], s['observed']
        lat.append(s['latency_seconds']); perr += s['parse_error']
        names = [n for b in BUCKETS for n in exp.get(b, [])]
        for n in names:
            eb = next(b for b in BUCKETS if n in exp.get(b, []))
            ob = next((b for b in BUCKETS if n in obs.get(b, [])), None)
            if eb == ob: sc += scripted; dr += not scripted
            if eb != 'executable' and ob == 'executable':
                adm += 1; admc += eb == 'blocked_critical'
    return dict(scripted=sc, derived=dr, total=sc + dr, admitted=adm, admitted_crit=admc, parse_errors=perr,
                mean_lat=st.mean(lat), median_lat=st.median(lat))
out = {}
for tag, f1 in RUN1.items():
    files = [os.path.join(EV, f1)] + sorted(glob.glob(os.path.join(HERE, f'{tag}_run*.json')))
    runs = [score(json.load(open(f, encoding='utf8'))) for f in files]
    out[tag] = runs
    agg = {k: (st.mean(r[k] for r in runs), min(r[k] for r in runs), max(r[k] for r in runs)) for k in runs[0]}
    print(f"{tag:14s} n={len(runs)} " + "  ".join(f"{k}={m:.1f}[{lo:.0f}-{hi:.0f}]" if 'lat' not in k else f"{k}={m:.2f}" for k, (m, lo, hi) in agg.items()))
json.dump(out, open(os.path.join(HERE, 'aggregate.json'), 'w'), indent=1)
