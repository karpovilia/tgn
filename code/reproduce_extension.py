"""Numerical replay of the E1/E2 prediction extension. No training or model loading.
Usage: python reproduce_extension.py --archive E1_E2_extension.zip
Dependencies: numpy. All AP calculations use tied score groups.
"""
import argparse,collections,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np

def ap(z):
 s=np.asarray(z,dtype=float).ravel();y=np.tile([1.,0.],len(z));i=np.argsort(-s,kind='stable');s=s[i];y=y[i]
 ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(y)[ends]
 return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/tp[-1])
def metrics(z):
 d=z[:,0]-z[:,1];return {'pairwise_accuracy':float(np.mean((d>0)+.5*(d==0))), 'pooled_AP':ap(z),'centered_AP':ap(z-z.mean(axis=1,keepdims=True))}
def close(a,b):
 assert abs(a-b)<5e-12,(a,b)
def run(path):
 with zipfile.ZipFile(path) as zip:
  manifest=json.loads(zip.read('MANIFEST.json'));checks=0;canonical=None;cache={}
  for name,m in manifest.items():
   raw=zip.read(name);assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256'],name
   if not name.endswith('.npz'):continue
   with np.load(io.BytesIO(raw),allow_pickle=False) as z:
    d={k:z[k].copy() for k in z.files}
   assert str(d['split'])=='test' and len(d['query_id'])==61763 and np.isfinite(d['logit']).all()
   if canonical is None:canonical={k:d[k] for k in ['query_id','src','timestamp','candidate_id']}
   for k,v in canonical.items():assert np.array_equal(d[k],v),(name,k)
   col=d['candidate_id'][:,0]==d['candidate_id'][:,1];assert col.sum()==670
   mm={pop:metrics(d['logit'][mask]) for pop,mask in [('all',np.ones(len(col),bool)),('no_candidate_collisions',~col)]}
   cache[name]={'metrics':mm}
   if name.startswith('E2/'):
    # Keep fixed-weight input signatures and logits for exposure replay.
    cache[name]['data']=d
   checks+=1
   if checks%32==0:print('verified prediction panels',checks,flush=True)
  assert checks==224
  e1=json.loads(zip.read('E1/report.json'));e2=json.loads(zip.read('E2/report.json'));seeds=list(range(1004,1010));perms=[42,10001,10002,10003,10004]
  for unit,expected in e1['panels'].items():
   v=cache['E1/predictions/'+unit+'/test.npz']['metrics']
   for pop in expected:
    for metric in expected[pop]:close(v[pop][metric],expected[pop][metric])
  for pop in ['all','no_candidate_collisions']:
   contrast={}
   for mode in ['on','off']:
    means=collections.defaultdict(list)
    for seed in seeds:
     rows=[]
     for perm in perms:
      def get(chron,h):return cache[f'E1/predictions/TCL_mooc_seed{seed}_{chron}_time_{mode}_test_history_{h}/test.npz']['metrics'][pop]
      cells=dict(A=get('native','native'),B=get('native',f'perm{perm}'),C=get(f'perm{perm}','native'),D=get(f'perm{perm}',f'perm{perm}'))
      row={}
      for metric in cells['A']:
       v={c:cells[c][metric] for c in cells}
       for c in cells:row['cells/'+metric+'/'+c]=v[c]
       for k,x in [('training_A-C',v['A']-v['C']),('history_A-B',v['A']-v['B']),('history_C-D',v['C']-v['D']),('interaction',v['A']-v['B']-v['C']+v['D'])]:row['effects/'+metric+'/'+k]=x
      rows.append(row)
     for k in rows[0]:means[k].append(float(np.mean([r[k] for r in rows])))
    for k,values in means.items():
     section,metric,key=k.split('/');ref=e1['results'][pop]['ON_OFF'][mode][section][metric][key];v=np.array(values);ci=2.570581835636314*v.std(ddof=1)/np.sqrt(6)
     for a,b in zip_values(values,ref['seed_values']):close(a,b)
     close(float(v.mean()),ref['mean']);close(float(v.mean()-ci),ref['CI95'][0]);close(float(v.mean()+ci),ref['CI95'][1])
    contrast[mode]=means
   for metric in ['pairwise_accuracy','pooled_AP','centered_AP']:
    v=np.array(contrast['on']['effects/'+metric+'/training_A-C'])-np.array(contrast['off']['effects/'+metric+'/training_A-C']);ref=e1['results'][pop]['delta_time_paired_seed_interval'][metric];ci=2.570581835636314*v.std(ddof=1)/np.sqrt(6)
    close(float(v.mean()),ref['mean']);close(float(v.mean()-ci),ref['CI95'][0]);close(float(v.mean()+ci),ref['CI95'][1])
  renamed={'accuracy':'pairwise_accuracy','AP':'pooled_AP','centered_AP':'centered_AP'};aggregates=collections.defaultdict(list)
  for row in e2['rows']:
   seed,perm=row['seed'],row['permutation'];pop='all' if row['population']=='all' else 'no_candidate_collisions'
   def get(chron,h):return cache[f'E2/predictions/DyGFormer_mooc_seed{seed}_{chron}/test_history_{h}.npz']
   cells=dict(A=get('native','native'),B=get('native',f'perm{perm}'),C=get(f'perm{perm}','native'),D=get(f'perm{perm}',f'perm{perm}'));effects={}
   for metric,original in renamed.items():
    vals={c:v['metrics'][pop][original] for c,v in cells.items()}
    for c,v in vals.items():close(v,row['cells'][c][metric]);effects[metric+'/'+c]=v
    for k,v in [('AC',vals['A']-vals['C']),('AB',vals['A']-vals['B']),('CD',vals['C']-vals['D']),('I',vals['A']-vals['B']-vals['C']+vals['D'])]:effects[metric+'/'+k]=v
   for label,l,r in [('AB',cells['A']['data'],cells['B']['data']),('CD',cells['C']['data'],cells['D']['data'])]:
    col=l['candidate_id'][:,0]==l['candidate_id'][:,1];mask=np.ones(len(col),bool) if pop=='all' else ~col;ex=(l['actual_input_sha256']!=r['actual_input_sha256'])[mask];zl=l['logit'][mask];zr=r['logit'][mask]
    flip=np.sign(zl[:,0]-zl[:,1])!=np.sign(zr[:,0]-zr[:,1]);assert not flip[~ex].any();assert np.all(np.abs(zl-zr)[~ex]<=(2e-5+1e-5*(np.abs(zl)+np.abs(zr)))[~ex])
    effects[label+'/exposure']=float(ex.mean());effects[label+'/flip_all']=float(flip.mean());effects[label+'/flip_exposed']=float(flip[ex].mean());effects[label+'/accuracy_exposed']=float(np.mean((zl[ex,0]>zl[ex,1])+.5*(zl[ex,0]==zl[ex,1]))-np.mean((zr[ex,0]>zr[ex,1])+.5*(zr[ex,0]==zr[ex,1])))
   for k,v in effects.items():close(v,row['effects'][k]);aggregates[row['population'],seed,k].append(v)
  for pop,ss in e2['summaries'].items():
   for k,ref in ss.items():
    v=[float(np.mean(aggregates[pop,s,k])) for s in [1004,1005]]
    for a,b in zip_values(v,ref['seed_values']):close(a,b)
    close(float(np.mean(v)),ref['mean']);close(min(v),ref['range'][0]);close(max(v),ref['range'][1])
  print(json.dumps({'status':'pass','prediction_panels':checks,'E1_models':72,'E2_models':12,'canonical_queries_identical':True,'candidate_collisions':670,'E1_seed_intervals_reproduced':True,'E2_two_seed_means_and_exposure_reproduced':True,'test_effects_not_used_for_selection':True}),flush=True)
# Avoid shadowing the builtin zip with the archive handle.
zip_values=zip
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);a=p.parse_args();run(a.archive)
