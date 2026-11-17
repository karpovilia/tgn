"""Actual model adapter. Imported ONLY inside an allocated remote worker."""
import copy
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from data_runtime import train_negatives,array_hash,get_neighbor_sampler
from protocol import atomic,MiB
from utils.utils import create_optimizer,set_random_seed
from models.modules import LinkPredictor_v1


def metrics(pos,neg,batch_size):
    pos=np.asarray(pos);neg=np.asarray(neg)
    if len(pos)!=len(neg) or not len(pos) or not np.isfinite(pos).all() or not np.isfinite(neg).all():raise ValueError("invalid prediction arrays")
    target=np.r_[np.ones(len(pos)),np.zeros(len(neg))];score=np.r_[pos,neg]
    ranks=1+(neg>pos)+.5*(neg==pos)
    batch_ap=[average_precision_score(np.r_[np.ones(len(pos[i:i+batch_size])),np.zeros(len(neg[i:i+batch_size]))],np.r_[pos[i:i+batch_size],neg[i:i+batch_size]]) for i in range(0,len(pos),batch_size)]
    return {"pooled_ap":float(average_precision_score(target,score)),"roc_auc":float(roc_auc_score(target,score)),"mrr":float(np.mean(1/ranks)),"hits_at_1":float(np.mean(ranks<=1)),"archival_mean_minibatch_ap":float(np.mean(batch_ap)),"queries":len(pos),"prediction_rows":2*len(pos),"negatives_per_query":1,"tie_policy":"average_rank"}


class Backend:
    def __init__(self,data,unit,cfg):
        self.data=data;self.unit=unit;self.cfg=cfg;self.args=cfg["args"];a=self.args
        self.seed=unit["training_seed"];self.name=unit["model"]
        set_random_seed(self.seed,deterministic_alg=True);torch.set_num_threads(1)
        self.samplers={k:get_neighbor_sampler(data=data[k],node_domain_size=len(data["node"]),sample_neighbor_strategy=a["sample_neighbor_strategy"],time_scaling_factor=a["time_scaling_factor"],seed=i) for k,i in [("train",0),("full",1)]}
        common=dict(node_raw_features=data["node"],edge_raw_features=data["edge"],neighbor_sampler=self.samplers["train"],time_feat_dim=a["time_feat_dim"],dropout=a["dropout"],device="cuda:0")
        if self.name=="TCL":
            from models.TCL import TCL
            backbone=TCL(**common,num_layers=a["num_layers"],num_heads=a["num_heads"],num_depths=a["num_neighbors"]+1)
        elif self.name=="CAWN":
            from models.CAWN import CAWN
            backbone=CAWN(**common,position_feat_dim=a["position_feat_dim"],walk_length=a["walk_length"],num_walk_heads=a["num_walk_heads"])
        elif self.name=="GraphMixer":
            from models.GraphMixer import GraphMixer
            backbone=GraphMixer(**common,num_tokens=a["num_neighbors"],num_layers=a["num_layers"])
        elif self.name=="DyGFormer":
            from models.DyGFormer import DyGFormer
            backbone=DyGFormer(**common,channel_embedding_dim=a["channel_embedding_dim"],patch_size=a["patch_size"],num_layers=a["num_layers"],num_heads=a["num_heads"],max_input_sequence_length=a["max_input_sequence_length"])
        elif self.name=="DyRep":
            from models.MemoryModel import MemoryModel,compute_src_dst_node_time_shifts
            t=data["train"];shifts=compute_src_dst_node_time_shifts(t.src_node_ids,t.dst_node_ids,t.node_interact_times)
            backbone=MemoryModel(**common,model_name="DyRep",num_layers=a["num_layers"],num_heads=a["num_heads"],src_node_mean_time_shift=shifts[0],src_node_std_time_shift=shifts[1],dst_node_mean_time_shift_dst=shifts[2],dst_node_std_time_shift=shifts[3],beta=a["pint_beta"],num_hop=a["pint_hop"])
        else:raise ValueError("unsupported model")
        self.model=torch.nn.Sequential(backbone,LinkPredictor_v1(172,172,172,1,random_projections=None,not_encode=False)).to("cuda:0")
        self.optimizer=create_optimizer(self.model,a["optimizer"],a["learning_rate"],a["weight_decay"])
        self.loss=torch.nn.BCEWithLogitsLoss();self.active_sampler="train";self.negative_descriptor=None
        self.batch_size=a["batch_size"];self.n_batches=(data["train"].num_interactions+self.batch_size-1)//self.batch_size

    def begin_epoch(self,epoch):
        self.model.train();self.model[0].set_neighbor_sampler(self.samplers["train"]);self.active_sampler="train"
        if self.name=="DyRep":self.model[0].memory_bank.__init_memory_bank__()
        self.negative_descriptor={"algorithm":"splitmix64_event_seed_epoch_v1","seed":self.seed,"epoch":epoch}

    def pair(self,split,start,negative):
        data=self.data[split];sl=slice(start,min(start+self.batch_size,data.num_interactions))
        src=data.src_node_ids[sl];dst=data.dst_node_ids[sl];ts=data.node_interact_times[sl];ids=data.edge_ids[sl]
        def embed(d,positive):
            kw=dict(src_node_ids=src,dst_node_ids=d,node_interact_times=ts)
            if self.name in ["TCL","CAWN","GraphMixer","DyRep"]:kw["num_neighbors"]=self.args["num_neighbors"]
            if self.name=="GraphMixer":kw["time_gap"]=self.args["time_gap"]
            if self.name=="DyRep":kw.update(edge_ids=ids if positive else None,edges_are_positive=positive)
            x,y=self.model[0].compute_src_dst_node_temporal_embeddings(**kw)
            return self.model[1](src,d,x,y).flatten()
        if self.name=="DyRep":
            neg=embed(negative,False);pos=embed(dst,True)
        else:pos=embed(dst,True);neg=embed(negative,False)
        return pos,neg

    def train_batch(self,index,epoch):
        self.model.train();start=index*self.batch_size
        ids=self.data["train"].edge_ids[start:start+self.batch_size]
        negative=train_negatives(ids,self.data["train_destinations"],self.seed,epoch)
        pos,neg=self.pair("train",start,negative)
        scores=torch.cat((pos,neg));target=torch.cat((torch.ones_like(pos),torch.zeros_like(neg)))
        loss=self.loss(scores,target)
        if not torch.isfinite(loss):raise RuntimeError("non-finite training loss")
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):raise RuntimeError("non-finite gradient")
        self.optimizer.step()
        if self.name=="DyRep":self.model[0].memory_bank.detach_memory_bank()
        return float(loss.detach().cpu())

    def evaluate(self,split,path):
        self.model.eval();self.model[0].set_neighbor_sampler(self.samplers["full"]);self.active_sampler="full"
        positives=[];negatives=[];data=self.data[split]
        with torch.no_grad():
            for start in range(0,data.num_interactions,self.batch_size):
                neg_ids=self.data["negatives"][split][start:start+self.batch_size]
                pos,neg=self.pair(split,start,neg_ids)
                positives.append(pos.cpu().numpy());negatives.append(neg.cpu().numpy())
        pos=np.concatenate(positives);neg=np.concatenate(negatives)
        report=metrics(pos,neg,self.batch_size)
        for label,code in [("transductive",0),("new_node",1)]:
            mask=self.data["cohort"][split]==code
            report[label]=metrics(pos[mask],neg[mask],self.batch_size) if mask.any() else None
        # One row per query with two candidate slots; lossless expanded schema in predictions.schema.json.
        atomic(path,lambda f:np.savez(f,schema_version=np.array("ctdg.predictions.v1"),run_id=np.array(self.unit["id"]),split=np.array(split),query_id=data.edge_ids,src=data.src_node_ids,timestamp=data.node_interact_times,candidate_id=np.column_stack((data.dst_node_ids,self.data["negatives"][split])),logit=np.column_stack((pos,neg)),target=np.tile(np.array([1,0],dtype=np.uint8),(len(pos),1)),cohort=self.data["cohort"][split]),8*MiB)
        return report

    def capture(self):
        torch.cuda.synchronize()
        state={"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_cpu_rng":torch.get_rng_state(),"torch_cuda_rng":torch.cuda.get_rng_state_all(),"samplers":{k:v.random_state.get_state() for k,v in self.samplers.items()},"active_sampler":self.active_sampler,"training_negative_schedule":self.negative_descriptor,"model_training":self.model.training,"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"memory":None}
        if self.name=="DyRep":state["memory"]=self.model[0].memory_bank.backup_memory_bank()
        return state

    def restore(self,state):
        self.model.load_state_dict(state["model"]);self.optimizer.load_state_dict(state["optimizer"])
        if self.name=="DyRep":self.model[0].memory_bank.reload_memory_bank(state["memory"])
        # Assign directly: set_neighbor_sampler would reset the saved sampler state.
        self.active_sampler=state["active_sampler"];self.model[0].neighbor_sampler=self.samplers[self.active_sampler]
        for k,s in state["samplers"].items():self.samplers[k].random_state.set_state(s)
        self.negative_descriptor=state["training_negative_schedule"];self.model.train(state["model_training"])
        torch.use_deterministic_algorithms(state["deterministic_algorithms"])
        random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_cpu_rng"].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda_rng"]])

    @staticmethod
    def save(state,f):torch.save(state,f)
    @staticmethod
    def load(path):return torch.load(path,map_location="cuda:0",weights_only=False)
