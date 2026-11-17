"""Frozen full-stream loading and condition-independent candidate schedule."""
import hashlib
from pathlib import Path
import numpy as np
from protocol import read


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def train_negatives(event_ids, destinations, seed, epoch):
    # Counter-based SplitMix64; mapping is by immutable event ID, not shuffled slot.
    # No global RNG consumption; no fitted/outcome-dependent candidate construction.
    x=np.asarray(event_ids,dtype=np.uint64) ^ np.uint64(seed<<32) ^ np.uint64(epoch<<16)
    with np.errstate(over="ignore"):
        x=x+np.uint64(0x9E3779B97F4A7C15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        x=x^(x>>np.uint64(31))
    return destinations[(x % np.uint64(len(destinations))).astype(np.int64)]


def load(package, unit, include_test=False):
    from utils.DataLoader import Data
    root=Path(package)/"data"/unit["dataset"]
    contract=read(root/"contract.json")
    arr=np.load(root/"events.npz",allow_pickle=False)
    src,dst,ts,ids,labels=[arr[k].copy() for k in ["src","dst","ts","event_id","label"]]
    if unit["permutation_seed"] is not None:
        order=np.load(root/("order_perm"+str(unit["permutation_seed"])+".npy"),allow_pickle=False)
        src,dst,ids,labels=[a[order] for a in [src,dst,ids,labels]]
    train=(ts<=contract["time_boundaries"][0]) & ~np.isin(src,contract["heldout_node_ids"]) & ~np.isin(dst,contract["heldout_node_ids"])
    val=(ts>contract["time_boundaries"][0]) & (ts<=contract["time_boundaries"][1]);test=ts>contract["time_boundaries"][1]
    def data(mask): return Data(src[mask],dst[mask],ts[mask],ids[mask],labels[mask])
    masks={"train":train,"val":val}
    if include_test:masks["test"]=test
    for name,mask in masks.items():
        if len(ids[mask])!=contract["counts"][name] or array_hash(np.sort(ids[mask]))!=contract["split_event_id_hashes"][name]:
            raise RuntimeError("frozen split identity changed: "+name)
    node=np.load(root/"node.npy",allow_pickle=False).astype(np.float32)
    edge=np.load(root/"edge.npy",allow_pickle=False).astype(np.float32)
    node=np.pad(node,((0,0),(0,172-node.shape[1])));edge=np.pad(edge,((0,0),(0,172-edge.shape[1])))
    train_nodes=set(src[train])|set(dst[train])
    cohort={name:np.array([0 if s in train_nodes and d in train_nodes else 1 for s,d in zip(src[mask],dst[mask])],dtype=np.uint8) for name,mask in masks.items() if name!="train"}
    return {"node":node,"edge":edge,"full":data(np.ones(len(ts),dtype=bool) if include_test else ts<=contract["time_boundaries"][1]),**{k:data(v) for k,v in masks.items()},
            "negatives":{k:np.load(root/("negative_"+k+".npy"),allow_pickle=False) for k in (["val","test"] if include_test else ["val"])},
            "cohort":cohort,"contract":contract,"train_destinations":np.unique(dst[train])}


def get_neighbor_sampler(data, node_domain_size, **kwargs):
    """Retain baseline sampler; represent valid yet-unobserved nodes by empty lists."""
    from utils.utils import get_neighbor_sampler as baseline_sampler
    sampler = baseline_sampler(data=data, **kwargs)
    size = len(sampler.nodes_neighbor_ids)
    if size > node_domain_size:
        raise ValueError('observed node lies outside frozen feature domain')
    for field in ['nodes_neighbor_ids', 'nodes_edge_ids', 'nodes_neighbor_times']:
        values = getattr(sampler, field)
        values.extend(np.array([]) for _ in range(node_domain_size - size))
    if hasattr(sampler, 'nodes_neighbor_sampled_probabilities'):
        sampler.nodes_neighbor_sampled_probabilities.extend(np.array([]) for _ in range(node_domain_size - size))
    return sampler
