"""Independent Full acceptance from raw saved arrays and coefficients; no fitting."""
from pathlib import Path
import csv,json,hashlib,resource,os
import numpy as np
from scipy.stats import t
from features import Features,reference
P=Path(__file__).resolve().parent
os.chdir(P)
OUT=P/"reproduced";OUT.mkdir(exist_ok=True)
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
def read(p):return json.loads(p.read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def acc(m):return float(np.mean((m>0)+.5*(m==0)))
def interval(v):
    v=np.asarray(v);m=float(v.mean());h=float(t.ppf(.975,9)*v.std(ddof=1)/np.sqrt(len(v)))
    return m,[m-h,m+h]

manifest=read(P/'MANIFEST.json')
for name,item in manifest.items():
    file=P/name;assert file.stat().st_size==item['bytes'] and sha(file)==item['sha256'],name
plan=read(P/'plan.json');result=read(P/'RESULT.json');ph=plan['scientific_plan_sha256']
assert ph==result['plan_sha256']
root=P/'reference';values={};group_rows=[];maxerror=0.;scalererror=0.;models=0;reused=0;new=0
for ds in ['mooc','wikipedia']:
    base=root/'logistic'/ds;contract=read(base/'inputs/contract.json');raw=load(base/'inputs/events.npz')
    a={k:raw[k] for k in ['src','dst','ts','event_id']};del raw
    vb,tb=contract['time_boundaries'];train=(a['ts']<=vb)&~np.isin(a['src'],contract['heldout_node_ids'])&~np.isin(a['dst'],contract['heldout_node_ids'])
    it=np.flatnonzero(train);iq=np.flatnonzero(a['ts']>tb);tau=float(np.median(np.diff(np.unique(a['ts'][train]))))
    native=Features(a['src'],a['dst'],a['ts'],train|(a['ts']>vb),tau)
    for entry in plan['draws'][ds]:
        s=entry['historical_draw'];data=load(P/f'data/{ds}_{s}.npz');draw=load(base/f'data/draw_{s}.npz');ev=load(base/f'data/eval_{s}.npz')
        assert np.array_equal(data['test_query_id'],a['event_id'][iq[ev['common']]])
        assert np.all(data['timestamp']>tb)
        assert np.all(data['positive']!=data['random_negative']) and np.all(data['positive']!=data['historical_negative'])
        fitted=it[draw['fit_eligible']];assert np.array_equal(data['train_query_id'],a['event_id'][fitted])
        sample=np.sort(np.random.default_rng(915).choice(len(data['test_query_id']),64,replace=False))
        for strategy in ['random','historical']:
            assert np.array_equal(reference(native,iq[ev['common']][sample],data[strategy+'_negative'][sample]),data[strategy+'_features'][sample,1])
        for cond in ['native','global']:
            ts=a['ts'].copy();ts[it[draw['order_'+cond]]]=a['ts'][it]
            f=Features(a['src'],a['dst'],ts,train,tau)
            train_features=np.stack([f.values(fitted,a['dst'][fitted]),f.values(fitted,data['train_negative'])],axis=1)
            trsample=np.sort(np.random.default_rng(916).choice(len(fitted),64,replace=False))
            assert np.array_equal(reference(f,fitted[trsample],data['train_negative'][trsample]),train_features[trsample,1])
            for strategy,key in [('random','random_features'),('historical','historical_features')]:
                x=data[key];ps=x[:,0,3]>0;ns=x[:,1,3]>0
                groups={'all':np.ones(len(x),bool),'both_seen':ps&ns,'positive_seen_negative_unseen':ps&~ns,
                        'positive_unseen_negative_seen':~ps&ns,'both_unseen':~ps&~ns}
                for variant,k in [('counts',6),('ages',9)]:
                    name=f'{ds}_{s}_{cond}_{variant}'
                    if strategy=='random':
                        folder=P/'fits'/name;record=read(folder/'result.json')
                        for file in ['model.npz','predictions.npz']:assert record['hashes'][file]==sha(folder/file)
                        assert record['converged'] and record['plan_sha256']==ph
                        model=load(folder/'model.npz');pred=load(folder/'predictions.npz');models+=1;new+=record['new_fit']
                        flat=train_features[:,:,:k].reshape(-1,k);mean=flat.mean(0);std=flat.std(0);std[std==0]=1
                        scalererror=max(scalererror,float(np.max(np.abs(mean-model['mean']))),float(np.max(np.abs(std-model['scale']))))
                        assert np.allclose(mean,model['mean'],atol=1e-12,rtol=1e-12) and np.allclose(std,model['scale'],atol=1e-12,rtol=1e-12)
                    else:
                        model=load(base/f'fits/{s}/{cond}/{variant}/model.npz');pred=load(P/'historical_test'/f'{name}.npz');reused+=1
                    assert np.array_equal(pred['query_id'],data['test_query_id'])
                    expected=(x[:,0,:k]-x[:,1,:k])@(model['coef'].ravel()/model['scale']);recorded=pred['logit'][:,0]-pred['logit'][:,1]
                    err=float(np.max(np.abs(expected-recorded)));maxerror=max(maxerror,err)
                    assert np.allclose(expected,recorded,atol=1e-11,rtol=1e-11) and np.array_equal(np.sign(expected),np.sign(recorded))
                    values[ds,strategy,s,cond,variant]=acc(recorded)
                    if strategy=='historical':
                        archived=read(base/f'fits/{s}/{cond}/{variant}/result.json')
                        assert abs(acc(recorded)-archived['matched_historical']['pairwise_accuracy'])<1e-14
                    for group,m in groups.items():
                        group_rows.append({'dataset':ds,'strategy':strategy,'draw':s,'condition':cond,'variant':variant,
                                           'group':group,'queries':int(m.sum()),'accuracy':acc(recorded[m]) if m.any() else None})
            del train_features,f
        print('VERIFIED_DRAW',ds,s,flush=True)
    del native,a
contrasts=[];table=[]
composition=read(P/'composition.json')
for ds in ['mooc','wikipedia']:
    bystrategy={}
    for strategy in ['historical','random']:
        rows=[]
        for entry in plan['draws'][ds]:
            s=entry['historical_draw'];a=values[ds,strategy,s,'native','ages']-values[ds,strategy,s,'global','ages'];c=values[ds,strategy,s,'native','counts']-values[ds,strategy,s,'global','counts']
            row={'dataset':ds,'strategy':strategy,'draw':s,'delta_ages':a,'delta_counts':c,'delta_recency':a-c};contrasts.append(row);rows.append(row)
        bystrategy[strategy]=rows
        table_row={'dataset':ds,'strategy':strategy}
        for key in ['delta_ages','delta_counts','delta_recency']:
            vals=[r[key] for r in rows];mean,ci=interval(vals);expected=result['summary'][ds][strategy][key]
            assert mean==expected['mean'] and ci==expected['CI95']
            table_row.update({key+'_pp':mean*100,key+'_CI_low_pp':ci[0]*100,key+'_CI_high_pp':ci[1]*100})
        table_row['positive_recency_draws']=sum(r['delta_recency']>0 for r in rows)
        co=[r for r in group_rows if r['dataset']==ds and r['strategy']==strategy and r['condition']=='native' and r['variant']=='ages']
        table_row['test_queries_min']=min(r['queries'] for r in co if r['group']=='all')
        table_row['test_queries_max']=max(r['queries'] for r in co if r['group']=='all')
        for g in ['both_seen','positive_seen_negative_unseen','positive_unseen_negative_seen','both_unseen']:
            table_row[g+'_queries_min']=min(r['queries'] for r in co if r['group']==g)
            table_row[g+'_queries_max']=max(r['queries'] for r in co if r['group']==g)
        table.append(table_row)
    paired=[r['delta_recency']-h['delta_recency'] for r,h in zip(bystrategy['random'],bystrategy['historical'])]
    mean,ci=interval(paired);assert mean==result['summary'][ds]['random_minus_historical_delta_recency']['mean']
    assert ci==result['summary'][ds]['random_minus_historical_delta_recency']['CI95']
assert contrasts==result['contrasts'] and models==80 and reused==80 and new==68
group_lookup={(r['dataset'],r['strategy'],r['draw'],r['condition'],r['variant'],r['group']):r for r in group_rows}
group_contrasts=[]
for ds in ['mooc','wikipedia']:
    for strategy in ['historical','random']:
        for entry in plan['draws'][ds]:
            s=entry['historical_draw']
            for group in ['all','both_seen','positive_seen_negative_unseen','positive_unseen_negative_seen','both_unseen']:
                values_for_group={}
                for variant in ['counts','ages']:
                    left=group_lookup[ds,strategy,s,'native',variant,group];right=group_lookup[ds,strategy,s,'global',variant,group]
                    assert left['queries']==right['queries']
                    values_for_group[variant]=left['accuracy']-right['accuracy'] if left['queries'] else None
                group_contrasts.append({'dataset':ds,'strategy':strategy,'draw':s,'group':group,'queries':left['queries'],
                                        'delta_ages':values_for_group['ages'],'delta_counts':values_for_group['counts'],
                                        'delta_recency':values_for_group['ages']-values_for_group['counts'] if left['queries'] else None})
for name,rows in [('table2.csv',table),('group_metrics.csv',group_rows),('per_draw_contrasts.csv',contrasts),('group_contrasts.csv',group_contrasts)]:
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
out={'status':'pass','random_models':models,'new_fits':new,'poc_models_reused':12,'historical_models':reused,
     'max_margin_reconstruction_error':maxerror,'max_train_scaler_reconstruction_error':scalererror,'ranking_mismatches':0,
     'all_prediction_ids_are_TEST':True,'all_candidates_noncolliding':True,'raw_event_causal_feature_spot_checks':'64 TRAIN and64 TEST queries per draw/chronology, equal-time exclusion',
     'all_primary_contrasts_and_CIs_exact':True,'all_historical_accuracy_values_match_frozen_baseline':True,'input_hashes_unchanged':True}
(OUT/'VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

for name in ['table2.csv','group_metrics.csv','group_contrasts.csv','per_draw_contrasts.csv']:
    assert (OUT/name).read_bytes()==(P/'recorded'/name).read_bytes(),name
print('PORTABLE_EXACT_REPRODUCTION_PASS')
