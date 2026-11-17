"""Synthetic forward/backward only: no dataset access, optimizer, or fitted model."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
package=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(package/'source'))
import torch
from models.DyGFormer import DyGFormer
from models.modules import LinkPredictor_v1
from utils.DataLoader import Data
from utils.utils import get_neighbor_sampler,set_random_seed

def run(seed):
 started=time.time();torch.set_num_threads(1)
 if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
 set_random_seed(seed,deterministic_alg=True)
 cfg=json.loads((package/'config.json').read_text())['args']
 assert cfg['model_name']=='DyGFormer' and cfg['batch_size']==200 and cfg['max_input_sequence_length']==256
 # Four frequently interacting nodes yield the full256-neighbor sequence shape.
 n=1600;src=np.tile(np.array([1,2,3,4]),n//4);dst=np.roll(src,1);ts=np.arange(1,n+1,dtype=float);ids=np.arange(1,n+1)
 history=Data(src,dst,ts,ids,np.ones(n));sampler=get_neighbor_sampler(history,sample_neighbor_strategy=cfg['sample_neighbor_strategy'],time_scaling_factor=cfg['time_scaling_factor'],seed=1)
 node=np.zeros((5,172),dtype=np.float32);edge=np.zeros((n+1,172),dtype=np.float32)
 backbone=DyGFormer(node_raw_features=node,edge_raw_features=edge,neighbor_sampler=sampler,time_feat_dim=cfg['time_feat_dim'],channel_embedding_dim=cfg['channel_embedding_dim'],patch_size=cfg['patch_size'],num_layers=cfg['num_layers'],num_heads=cfg['num_heads'],dropout=cfg['dropout'],max_input_sequence_length=cfg['max_input_sequence_length'],device='cuda:0')
 model=torch.nn.Sequential(backbone,LinkPredictor_v1(172,172,172,1,random_projections=None,not_encode=False)).to('cuda:0')
 qsrc=np.tile(np.array([1,2,3,4]),50);qdst=np.roll(qsrc,1);qt=np.full(200,n+10.,dtype=float)
 model.train();x,y=model[0].compute_src_dst_node_temporal_embeddings(qsrc,qdst,qt);score=model[1](qsrc,qdst,x,y).flatten()
 loss=torch.nn.functional.binary_cross_entropy_with_logits(score,torch.ones_like(score));loss.backward();torch.cuda.synchronize()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 return dict(status='pass',seed=seed,batch_size=cfg['batch_size'],max_sequence_length=cfg['max_input_sequence_length'],architecture={k:cfg[k] for k in ['time_feat_dim','channel_embedding_dim','patch_size','num_layers','num_heads','dropout']},config_sha256=hashlib.sha256((package/'config.json').read_bytes()).hexdigest(),source_sha256={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),package/'source/models/DyGFormer.py',package/'source/models/modules.py']},forward_backward=True,optimizer_steps=0,dataset_access=False,TEST_access=False,scientific_fits=0,
             peak_torch_GPU_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-started,python=sys.executable,torch=torch.__version__)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.seed);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
