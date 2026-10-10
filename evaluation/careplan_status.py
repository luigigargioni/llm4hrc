import os, json, glob, collections, re
CARE = {'physical exercises','read a book','take ibuprofen','take vitamin d','walk in the park'}
files=(sorted(glob.glob(os.environ['REPLAY_FILES'])) if os.environ.get('REPLAY_FILES') else [f for f in glob.glob('evaluation/archive/replay_pre_rev3/final_P*.json') if not any(x in f for x in ('P03b','crashed'))])
R=[r for f in files for r in json.load(open(f,encoding='utf8'))['transcripts']]
print('sessions',len(R))
def kind(v):
    rs=v['reasons']; s=v['arbitrated_status']
    if s=='EXECUTABLE': return 'EXEC'
    k=set()
    for x in rs:
        if x.startswith('unknown'): k.add('unknown')
        elif x.startswith('missing_precondition'): k.add('pre')
        elif 'expired' in x: k.add('expired')
        elif 'early' in x: k.add('early')
        else: k.add(re.split('[:]',x)[0])
    return '+'.join(sorted(k))
allk=collections.Counter(); care=collections.Counter(); carestat=collections.Counter(); pre=collections.Counter()
nonexec=0
for r in R:
    for v in r['validations']:
        k=kind(v); allk[k]+=1
        if v['activity_name'].strip().lower() in CARE:
            care[k]+=1; carestat[v['arbitrated_status']]+=1
            for x in v['reasons']:
                if x.startswith('missing_precondition'): pre[(v['canonical_action'],x.split(':',1)[1].strip())]+=1
N=sum(allk.values()); print('all decisions',N)
for k,c in allk.most_common(): print(f'  {k:25s} {c:5d} {c/N:.1%}')
M=sum(care.values()); print('care-plan decisions',M, f'{M/N:.1%}')
for k,c in care.most_common(): print(f'  {k:25s} {c:5d} {c/M:.1%}')
print(carestat)
print('missing preconditions on care-plan activities:'); [print('  ',k,c) for k,c in pre.most_common(15)]
