"""Frozen causal feature implementation extracted from prior validated source."""

from pathlib import Path

import hashlib,json,os

import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.writing');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)

def npz(p,**arrays):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.with_suffix('.writing').open('wb') as f:np.savez_compressed(f,**arrays)
 os.replace(p.with_suffix('.writing'),p)

def mix(ids,seed):
 x=np.asarray(ids,dtype=np.uint64)^np.uint64(seed<<32)^np.uint64(1<<16)
 with np.errstate(over='ignore'):
  x=x+np.uint64(0x9E3779B97F4A7C15);x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);x=x^(x>>np.uint64(31))
 return x

class Prefix:
 def __init__(self,key,ts):
  order=np.lexsort((ts,key));self.key=key[order];self.ts=ts[order];self.index=np.empty(len(key),dtype=[('key','<i8'),('ts','<f8')]);self.index['key']=self.key;self.index['ts']=self.ts
 def lookup(self,key,ts):
  if not len(self.key):return np.zeros(len(key),dtype=np.int64),np.zeros(len(key))
  q=np.empty(len(key),dtype=self.index.dtype);q['key']=key;q['ts']=ts
  pos=np.searchsorted(self.index,q,side='left');start=np.searchsorted(self.key,key,side='left');count=pos-start;assert np.all(count>=0)
  last=np.where(count>0,self.ts[np.maximum(pos-1,0)],0.);assert np.all(last[count>0]<ts[count>0]);return count,last

class Features:
 def __init__(self,src,dst,ts,history_mask,tau):
  self.src=src;self.dst=dst;self.ts=ts;self.hm=history_mask;self.tau=float(tau);self.base=int(max(src.max(),dst.max()))+1;assert self.base*self.base<2**63
  self.pair=Prefix(src[history_mask]*self.base+dst[history_mask],ts[history_mask]);self.source=Prefix(src[history_mask],ts[history_mask]);self.destination=Prefix(dst[history_mask],ts[history_mask])
 def values(self,indices,candidates):
  source=self.src[indices];ts=self.ts[indices];counts=[];ages=[]
  for index,keys in [(self.pair,source*self.base+candidates),(self.source,source),(self.destination,candidates)]:
   c,last=index.lookup(keys,ts);seen=c>0;counts.append(c);ages.append(np.where(seen,np.log1p(np.maximum(0,ts-last)/self.tau),0.))
  c=np.column_stack(counts);return np.column_stack([np.log1p(c),c>0,np.column_stack(ages)]).astype(np.float64)
 def hard(self,indices,ids,seed):
  keys,first=np.unique(self.pair.key,return_index=True);sources=keys//self.base;destinations=keys%self.base;first_times=self.pair.ts[first];u,starts,counts=np.unique(sources,return_index=True,return_counts=True);groups={int(s):(destinations[a:a+n],first_times[a:a+n]) for s,a,n in zip(u,starts,counts)}
  candidates=np.full(len(indices),-1,dtype=np.int64);available=np.zeros(len(indices),dtype=np.int64);hashes=mix(ids[indices],seed)
  for j,i in enumerate(indices):
   group=groups.get(int(self.src[i]))
   if group is None:continue
   d,t=group;allowed=d[(t<self.ts[i])&(d!=self.dst[i])];available[j]=len(allowed)
   if len(allowed):candidates[j]=allowed[int(hashes[j]%np.uint64(len(allowed)))]
  return candidates,available

def reference(features,indices,candidates):
 result=[]
 for i,dst in zip(indices,candidates):
  history=features.hm&(features.ts<features.ts[i]);masks=[history&(features.src==features.src[i])&(features.dst==dst),history&(features.src==features.src[i]),history&(features.dst==dst)];c=[int(m.sum()) for m in masks];age=[np.log1p((features.ts[i]-features.ts[m].max())/features.tau) if m.any() else 0. for m in masks];result.append(list(np.log1p(c))+[float(x>0) for x in c]+age)
 return np.asarray(result)
