import os, json, glob, re, csv, collections
from pathlib import Path
U = Path(r"C:/Users/Gargioni/Il mio Drive/Academic/_Unibs/PhD/Hybrid Architecture/User Study/Users")
CARE = {'physical exercises','read a book','take ibuprofen','take vitamin d','walk in the park'}
conv = {p.name.replace('_conversation.txt',''): p for p in U.glob('*/*_conversation.txt')}
LINE = re.compile(r'^(\d\d:\d\d) - (Patient|Perception|Robot \(speaking\)|Robot \(acting\)): ?(.*)$')
def parse_conv(p):
    turns = []  # list of dicts per time with patient, perception, robot speech, action
    cur = None
    # Some transcripts are UTF-8, others cp1252: try UTF-8 first.
    try:
        text = p.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = p.read_text(encoding='cp1252', errors='replace')
    for raw in text.splitlines():
        m = LINE.match(raw.strip())
        if not m:
            if cur and cur.get('_last') and raw.strip(): cur[cur['_last']] += ' ' + raw.strip()
            continue
        t, who, text = m.groups()
        if who == 'Patient' or cur is None or (who == 'Perception' and cur.get('perception')):
            cur = {'time': t, 'patient': '', 'perception': '', 'speech': '', 'action': ''}; turns.append(cur)
        key = {'Patient': 'patient', 'Perception': 'perception', 'Robot (speaking)': 'speech', 'Robot (acting)': 'action'}[who]
        if key in ('speech','action') and cur[key]:  # second reply in same turn
            cur = dict(cur, speech='', action=''); turns.append(cur)
        cur[key] = text; cur['_last'] = key
    return turns
files = (sorted(glob.glob(os.environ['REPLAY_FILES'])) if os.environ.get('REPLAY_FILES') else [f for f in glob.glob('evaluation/archive/replay_pre_rev3/final_P*.json') if not any(x in f for x in ('P03b','crashed'))])
rows = []; cyc_total = 0
for f in files:
    for r in json.load(open(f, encoding='utf8'))['transcripts']:
        sess = r['transcript'].split('/')[1]
        E = [json.loads(l) for l in open(r['structured_log'], encoding='utf8')]
        cycles = []; cur = None; prev = None
        for e in E:
            dt = e.get('decision_type')
            if dt == 'canonicalization' and prev != 'canonicalization':
                cur = {'time': None, 'arb': []}; cycles.append(cur)
            if cur is None: prev = dt; continue
            if dt == 'temporal_check' and cur['time'] is None: cur['time'] = e['current_time']
            if dt == 'arbitration': cur['arb'].append(e)
            prev = dt
        orig = parse_conv(conv[sess]) if sess in conv else []
        by_time = collections.defaultdict(list)
        for t in orig: by_time[t['time']].append(t)
        cyc_total += len(cycles)
        for c in cycles:
            for a in c['arb']:
                if a['activity_name'].strip().lower() not in CARE: continue
                miss = [x.split(':',1)[1].strip() for x in a['not_executable_reasons'] if x.startswith('missing_precondition')]
                if not miss: continue
                for o in by_time.get(c['time'], [])[:1]:
                    rows.append({'session': r['transcript'], 'time': c['time'], 'activity': a['activity_name'],
                        'status': a['arbitrated_status'], 'reasons': ';'.join(a['not_executable_reasons']),
                        'patient': o['patient'], 'perception': o['perception'], 'orig_action': o['action'], 'orig_speech': o['speech']})
# dedupe identical (session,time,activity)
seen=set(); out=[]
for x in rows:
    k=(x['session'],x['time'],x['activity'])
    if k in seen: continue
    seen.add(k); out.append(x)
print('cycles', cyc_total, 'candidate rows', len(rows), 'unique', len(out))
with open(os.environ.get("CANDIDATES_CSV", "evaluation/original_vs_layer_candidates.csv"), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
