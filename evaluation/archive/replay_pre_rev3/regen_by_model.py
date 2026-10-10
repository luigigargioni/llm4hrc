import json, glob, collections
CARE = {'physical exercises','read a book','take ibuprofen','take vitamin d','walk in the park'}
def load(files):
    out = {}
    for f in files:
        d = json.load(open(f, encoding='utf8'))
        for r in d['transcripts']:
            out[r['transcript'].split('/')[1]] = (r, d.get('model'))
    return out
oss = load([f for f in glob.glob('evaluation/archive/replay_pre_rev3/final_P*.json') if not any(x in f for x in ('P03b','crashed','P03a'))] + ['evaluation/archive/replay_pre_rev3/final_P03a.json'])
mini = load(['evaluation/archive/replay_pre_rev3/replay_results_full.json'])
print('sessions oss', len(oss), 'mini', len(mini))
def stats(r):
    names = {c['activity_name'] for c in r['canonicalizations']}
    intro = {n for n in names if n.strip().lower() not in CARE}
    unk = collections.defaultdict(set)
    for c in r['canonicalizations']: unk[c['activity_name']].add(c['canonical_action'])
    never = {n for n in intro if not (unk[n] - {'unknown', None})}
    return len(names), len(intro), len(never), r['turns_replayed'], r['outcome']
tot = collections.Counter()
for s in sorted(set(oss) & set(mini)):
    a, b = stats(mini[s][0]), stats(oss[s][0])
    print(s, 'mini names/intro/never/turns', a, '| oss', b)
    tot['mini_intro'] += a[1]; tot['oss_intro'] += b[1]; tot['mini_never'] += a[2]; tot['oss_never'] += b[2]; tot['n'] += 1
print(dict(tot))
print('---- complete in both')
both = [s for s in set(oss)&set(mini) if mini[s][0]['outcome']=='transcript_exhausted']
t=collections.Counter()
for s in sorted(both):
    rm=mini[s][0]; ro=oss[s][0]
    nm={c['activity_name'] for c in rm['canonicalizations'] if c['activity_name'].strip().lower() not in CARE}
    print(s, rm['transcript'], sorted(nm))
    a,b=stats(rm),stats(ro); t['mi']+=a[1]; t['oi']+=b[1]; t['mn']+=a[2]; t['on']+=b[2]; t['turns']+=a[3]
    t['mdec']+=len(rm['validations']); t['odec']+=len(ro['validations'])
    for key,r in (('m',rm),('o',ro)):
        t[key+'unk']+=sum(1 for v in r['validations'] if v['canonical_action'] in (None,'unknown'))
print(dict(t))
