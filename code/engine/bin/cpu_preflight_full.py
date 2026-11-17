"""Exact-launcher CPU imports, environment parity and frozen split identities."""
import argparse,json,os,sys
from pathlib import Path
package=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(package/'source'))
from environment import capture,guard
from protocol import read,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--expected',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--freeze',action='store_true');a=p.parse_args()
 assert os.environ['CUDA_VISIBLE_DEVICES']==''
 a.evidence.mkdir(parents=True,exist_ok=True)
 if a.freeze:
  assert not a.expected.exists();a.expected.write_text(json.dumps(capture(),indent=2)+'\n')
 guard(a.expected,a.evidence/'matching')
 bad=read(a.expected);bad['python']='deliberate mismatch';badpath=a.evidence/'wrong_environment.json';badpath.write_text(json.dumps(bad))
 try:guard(badpath,a.evidence/'mismatch')
 except RuntimeError:pass
 else:raise AssertionError('environment guard accepted mismatch')
 from full_backend import FullBackend
 from full_engine import run_segment
 from data_runtime import load
 import torch
 units=read(package/'units.json');native=load(package,units[0]);assert 'test' not in native and 'test' not in native['negatives']
 assert native['train'].num_interactions==227485 and native['val'].num_interactions==61762
 assert len(units)==36 and sum(u['reuse_POC'] for u in units)==6
 for perm in [42,10001,10002,10003,10004]:
  u=next(u for u in units if u['training_seed']==1004 and u['permutation_seed']==perm)
  d=load(package,u);assert d['val'].num_interactions==61762
 assert not torch.cuda.is_initialized()
 result=dict(status='pass',unit_count=36,POC_reuse=6,new_fits=30,guard_match='pass',guard_mismatch='pass',all5permutation_contracts='pass',GPU_initialized=False,model_instantiated=False,TEST_split_materialized=False,TEST_candidates_loaded=False,TEST_predictions=False)
 (a.evidence/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
