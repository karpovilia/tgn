"""Read-only replay of the two complete Tesla seed blocks; never dispatches jobs."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
from sklearn.metrics import average_precision_score
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/data/CTDG-revolt/wsdm_revision_E2_full_20261117/backup_tesla/tree/full')
read=lambda p:json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
scope=read(HERE/'evidence/E2_SCOPE.json');prepared=ROOT/'prepared';units=[u for u in read(prepared/'units.json') if u['training_seed'] in scope['seeds']]
assert len(units)==12
binding=read(prepared/'DATA_BINDING.json')
for name in ['contract.json','events.npz','negative_test.npy']:
 assert sha(Path(binding[name]['source']))==binding[name]['sha256']
contract=read(Path(binding['contract.json']['source']))
with np.load(binding['events.npz']['source'],allow_pickle=False) as z:
 mask=z['ts']>contract['time_boundaries'][1]
 canonical={'query_id':z['event_id'][mask], 'src':z['src'][mask], 'timestamp':z['ts'][mask], 'candidate_id':np.column_stack([z['dst'][mask],np.load(binding['negative_test.npy']['source'],allow_pickle=False)])}
assert len(canonical['query_id'])==61763
collision=canonical['candidate_id'][:,0]==canonical['candidate_id'][:,1];assert collision.sum()==670
panels={};provenance={};max_ap_diff=0.
def ap(scores):
 global max_ap_diff
 s=np.asarray(scores,dtype=float).ravel();y=np.tile([1,0],len(scores));idx=np.argsort(-s,kind='stable');ss=s[idx];yy=y[idx]
 ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(s)-1];tp=np.cumsum(yy)[ends];delta=np.diff(np.r_[0,tp]);v=float(np.sum(delta*(tp/(ends+1)))/tp[-1]);ref=average_precision_score(y,s);max_ap_diff=max(max_ap_diff,abs(v-ref));assert abs(v-ref)<5e-12;return v

def met(z):
 # Preserve production float32 centering, independently calculate AP tie groups.
 dif=z[:,0]-z[:,1]
 return {'accuracy':float(np.mean((dif>0)+.5*(dif==0))), 'AP':ap(z), 'centered_AP':ap(z-z.mean(axis=1,keepdims=True))}
for u in units:
 d=ROOT/'runtime'/u['id'];receipt=read(d/'accepted_full.json');assert receipt['status']=='complete'
 assert receipt['identity']=={'manifest_sha256':sha(prepared/'PACKAGE_MANIFEST.json'),'plan_sha256':sha(prepared/'PLAN.json'),'unit':u}
 for name,meta in receipt['files'].items():
  f=d/name;assert f.resolve().is_relative_to(d.resolve()) and not f.is_symlink();assert f.stat().st_size==meta['bytes'] and sha(f)==meta['sha256'],str(f)
 result=read(d/'result.json');assert result['status']=='complete' and result['test_inference_performed'] and result['split']=='test';assert 20<=result['completed_epochs']<=100
 refs=read(d/'refs.json')
 for role in ['latest','best']:
  m=refs[role];assert receipt['files'][m['path']]['sha256']==m['sha256']
 provenance[u['id']]={'receipt_sha256':sha(d/'accepted_full.json'),'epochs':result['completed_epochs'],'selected_epoch':result['early']['best_epoch'],'files':receipt['files'],'identity':receipt['identity']}
 files=list((d/'predictions').glob('test_history_*.npz'));assert len(files)==(6 if u['permutation_seed'] is None else 2)
 for f in files:
  assert str(f.relative_to(d)) in receipt['files']
  with np.load(f,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
  for k,v in canonical.items():assert np.array_equal(data[k],v),(f,k)
  assert str(data['split'])=='test' and np.isfinite(data['logit']).all()
  assert np.max(np.abs(data['logit'][collision,0]-data['logit'][collision,1]))<2e-5
  panels[u['training_seed'],u['permutation_seed'],str(data['history'])]=data
assert len(panels)==32
rows=[];quality=[];summaries={}
for pop,mask in [('all',np.ones(61763,dtype=bool)),('no_collisions',~collision)]:
 per_seed=[]
 for seed in scope['seeds']:
  sr=[];A=panels[seed,None,'native']
  for perm in scope['permutations']:
   h='perm'+str(perm);cells={'A':A,'B':panels[seed,None,h],'C':panels[seed,perm,'native'],'D':panels[seed,perm,h]};scores={c:met(d['logit'][mask]) for c,d in cells.items()};effects={}
   for metric in ['accuracy','AP','centered_AP']:
    v={c:scores[c][metric] for c in cells}
    for c in cells:effects[metric+'/'+c]=v[c]
    for name,value in [('AC',v['A']-v['C']),('AB',v['A']-v['B']),('CD',v['C']-v['D']),('I',(v['A']-v['B'])-(v['C']-v['D']))]:effects[metric+'/'+name]=value
   for name,l,r in [('AB',cells['A'],cells['B']),('CD',cells['C'],cells['D'])]:
    ex=(l['actual_input_sha256']!=r['actual_input_sha256'])[mask];tr=(l['transformer_input_sha256']!=r['transformer_input_sha256'])[mask];zl=l['logit'][mask];zr=r['logit'][mask]
    flip=np.sign(zl[:,0]-zl[:,1])!=np.sign(zr[:,0]-zr[:,1]);drift=np.abs(zl-zr);tol=2e-5+1e-5*(np.abs(zl)+np.abs(zr));assert np.all(drift[~ex]<=tol[~ex]);assert not flip[~ex].any()
    effects[name+'/exposure']=float(ex.mean());effects[name+'/flip_all']=float(flip.mean());effects[name+'/flip_exposed']=float(flip[ex].mean()) if ex.any() else None
    effects[name+'/accuracy_exposed']=met(zl[ex])['accuracy']-met(zr[ex])['accuracy'] if ex.any() else None
    quality.append({'population':pop,'seed':seed,'permutation':perm,'contrast':name,'exposed':int(ex.sum()),'unchanged':int((~ex).sum()),'unchanged_flips':int(flip[~ex].sum()),'unchanged_max_drift':float(drift[~ex].max()) if (~ex).any() else 0.,'transformer_exposed':int(tr.sum())})
   sr.append(effects);rows.append({'population':pop,'seed':seed,'permutation':perm,'cells':scores,'effects':effects})
  per_seed.append({k:float(np.mean([r[k] for r in sr])) if all(r[k] is not None for r in sr) else None for k in sr[0]})
 summaries[pop]={k:{'mean':float(np.mean(v)) if all(x is not None for x in v) else None,'seed_values':v,'range':[min(v),max(v)] if all(x is not None for x in v) else None} for k in per_seed[0] for v in [[s[k] for s in per_seed]]}
report={'status':'pass','scope':scope,'models':12,'unique_panels':32,'logical_AD_cells':40,'queries':61763,'candidate_collisions':670,'metrics':'logits; paired ties half credit; independent tied-score AP verified against sklearn','maximum_AP_replay_difference':max_ap_diff,'canonical_TCL_candidates_verified':True,'selection':'Owner-selected complete hardware blocks, not outcome-selected','summaries':summaries,'rows':rows,'numerical_audit':quality,'provenance':provenance}
(HERE/'evidence/E2_TWO_SEED_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
for k,v in summaries['all'].items():
 if k.startswith('accuracy/') or 'exposure' in k or 'flip' in k:print(k,v)
print('PASS',max_ap_diff,flush=True)
