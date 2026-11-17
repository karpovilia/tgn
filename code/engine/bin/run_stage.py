"""Full stage entry: frozen GO certificate, approved host binding, immutable inputs."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--unit',required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--execution-binding',type=Path,required=True);a=p.parse_args()
 package=Path(__file__).resolve().parent.parent;root=package.parent
 from protocol import read,digest,lock,write,StateStore
 binding=read(a.execution_binding)
 assert binding['scope']=='E2_FULL_36_DyGFormer_MOOC' and binding['approved'] is True and binding['owner_confirmation']
 start_fd=int(os.environ['CTDG_WORKER_START_FD']);assert os.read(start_fd,1)==b'1';os.close(start_fd)
 assert binding['cpu_threads']==1 and binding['paid_api_cost_cap_usd']==0
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==binding['gpu_uuid']
 assert os.path.ismount(binding['required_runtime_mount'])
 assert digest(root/'PLAN.json')==binding['full_plan_sha256']
 assert digest(root/'PACKAGE_MANIFEST.json')==binding['package_manifest_sha256']
 for name,meta in read(root/'PACKAGE_MANIFEST.json')['files'].items():
  assert digest(root/name)==meta['sha256'],name
 gate_path=Path(binding['poc_gate_receipt']);assert digest(gate_path)==binding['poc_gate_receipt_sha256']
 gate=read(gate_path);assert gate['status']=='GO' and gate['quality_gate'] is True and gate['TEST_access'] is False
 plan=read(root/'PLAN.json');assert digest(package/'config.json')==plan['training']['config_sha256']
 units=read(package/'units.json');unit=next(u for u in units if u['id']==a.unit)
 assert a.unit in binding['unit_ids']
 assert binding['hardware_family']==unit['within_seed_hardware_group'], 'Within-seed hardware family changed'
 from gpu_guard import verify_supervisor_lease,initialize_cuda,birth
 verify_supervisor_lease(binding)
 sys.path.insert(0,str(package/'source'))
 from environment import guard
 assert digest(Path(binding['expected_environment']))==binding['expected_environment_sha256']
 guard(binding['expected_environment'],a.state/'environment_checks'/a.unit)
 import torch
 from data_runtime import load
 from full_backend import FullBackend
 from full_engine import run_segment
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 initialize_cuda(binding,torch)
 identity={'plan_sha256':digest(root/'PLAN.json'),'manifest_sha256':digest(root/'PACKAGE_MANIFEST.json'),'unit':unit}
 out=a.state/a.unit
 with lock(out/'worker.lock'):
  data=load(package,unit);cfg=read(package/'config.json');backend=FullBackend(data,unit,cfg,package,plan['permutation_seeds']);store=StateStore(out,identity)
  parent=int(os.environ['CTDG_SUPERVISOR_PID']);parent_birth=birth(parent)
  deadline=float(os.environ['CTDG_BATCH_RUNTIME_DEADLINE'])
  def runtime_guard():
   if os.getppid()!=parent or birth(parent)!=parent_birth:raise OSError('Supervisor disappeared; stop before next model batch')
   if time.time()>=deadline:raise RuntimeError('Original cumulative runtime budget exhausted')
  backend.runtime_guard=runtime_guard
  if unit['reuse_POC'] and not store.refs():
   source=gate['receipts'][unit['id']];checkpoint=Path(source['selected_checkpoint'])
   assert digest(checkpoint)==source['selected_checkpoint_sha256']
   source_root=checkpoint.parent;accepted=read(source_root/'accepted_poc.json')
   assert digest(source_root/'accepted_poc.json')==source['acceptance_sha256']
   assert accepted['identity']['manifest_sha256']==plan['poc']['current_training_manifest_sha256']
   saved=backend.load(checkpoint);backend.restore(saved['backend'])
   old=read(source_root/'result.json');assert old['test_inference_performed'] is False and 20<=old['completed_epochs']<=100
   state={k:v for k,v in old.items() if k not in ['status','split','test_inference_performed','crossed_validation']}
   state.update(identity=identity,phase='crossed_test',reused_POC=True,backend=backend.capture())
   write(out/'reused_POC.json',dict(source=source,source_receipt=accepted,training_reperformed=False))
   store.save(state,backend.save,best=True)
  return run_segment(backend,store,cfg)
if __name__=='__main__':
 try:raise SystemExit(main())
 except (OSError,ConnectionError) as e:
  print('Infrastructure unavailable; retain pending work:',repr(e),file=sys.stderr);raise SystemExit(76)
