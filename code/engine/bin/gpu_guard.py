"""Whole-batch GPU lease and namespace-aware, serialized CUDA admission.

No foreign process is ever signalled. NVML host PIDs are associated with verified
pod PID births only by a singleton delta during serialized CUDA initialization.
The mapping protocol requires a separate successful target qualification.
"""
from pathlib import Path
import contextlib,fcntl,json,os,subprocess,time
from protocol import read,write,lock


def birth(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return {'parent_pid':int(fields[1]),'start_ticks':int(fields[19])}
 except (FileNotFoundError,ProcessLookupError):return None

def gpu_pids(grant):
 r=subprocess.run(['nvidia-smi','--id='+grant['gpu_uuid'],'--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
 if r.returncode:raise OSError('GPU inventory unavailable: '+r.stderr[-300:])
 return {int(row.strip()) for row in r.stdout.splitlines() if row.strip()}

@contextlib.contextmanager
def lease(grant):
 p=Path(grant['gpu_lease_path']);assert p.is_absolute() and grant['gpu_uuid'] in p.name
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);yield handle.fileno()

def validate_registry(registry,parent=None):
 parent=registry['supervisor_pid'] if parent is None else parent
 assert parent==registry['supervisor_pid']
 actual=birth(parent);assert actual and actual['start_ticks']==registry['supervisor_start_ticks']
 allowed=set()
 for pid,item in registry['workers'].items():
  actual=birth(int(pid))
  if actual and actual['parent_pid']==parent and actual['start_ticks']==item['start_ticks'] and item.get('host_gpu_pid') is not None:
   allowed.add(item['host_gpu_pid'])
 return allowed

def occupied(grant):
 registry=read(Path(grant['runtime_root'])/'active_workers.json')
 return bool(gpu_pids(grant)-validate_registry(registry))

def verify_supervisor_lease(grant):
 parent=int(os.environ['CTDG_SUPERVISOR_PID']);descriptor=int(os.environ['CTDG_GPU_LEASE_FD'])
 assert parent==os.getppid()
 actual=os.fstat(descriptor);expected=os.stat(grant['gpu_lease_path']);parent_fd=os.stat(f'/proc/{parent}/fd/{descriptor}')
 assert (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino)==(parent_fd.st_dev,parent_fd.st_ino)
 token='%02x:%02x:%d'%(os.major(actual.st_dev),os.minor(actual.st_dev),actual.st_ino)
 proc_owned=any(len(row)>5 and row[1:4]==['FLOCK','ADVISORY','WRITE'] and row[4]==str(parent) and row[5]==token for row in (line.split() for line in Path('/proc/locks').read_text().splitlines()))
 if not proc_owned:
  assert grant.get('lease_verification_mode')=='qualified_inherited_fd', 'Parent lock row is not visible; unqualified fallback forbidden'
  # This is the SAME open-file description explicitly inherited from our
  # verified live parent. Reassertion preserves its lifetime and never unlocks it.
  fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with Path(grant['gpu_lease_path']).open('a') as competitor:
   try:fcntl.flock(competitor,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:
    raise RuntimeError('Independent descriptor unexpectedly acquired inherited GPU lease')
 registry=read(Path(grant['runtime_root'])/'active_workers.json');validate_registry(registry,parent)
 item=registry['workers'][str(os.getpid())];assert item['start_ticks']==birth(os.getpid())['start_ticks']
 known=validate_registry(registry,parent)
 assert not (gpu_pids(grant)-known), 'Unmapped GPU process before CUDA initialization'

def initialize_cuda(grant,torch):
 """Called once per worker; parent permits only one uninitialized worker at once."""
 state=Path(grant['runtime_root']);registry_path=state/'active_workers.json'
 with lock(state/'cuda_initialization.lock'):
  verify_supervisor_lease(grant)
  # Capture verified ownership before the GPU sample: a known peer may exit
  # during inventory and disappear from the registry before CUDA is created.
  registry_before=read(registry_path)
  known_before=validate_registry(registry_before,int(os.environ['CTDG_SUPERVISOR_PID']))
  before=gpu_pids(grant)
  assert before.issubset(known_before), 'Unmapped GPU process in CUDA baseline snapshot'
  torch.cuda.set_per_process_memory_fraction(grant['gpu_memory_fraction_per_lane'],device=0)
  marker=torch.empty(1,device='cuda:0');torch.cuda.synchronize()
  end=time.monotonic()+10
  while True:
   after=gpu_pids(grant);added=after-before
   # Previously verified own peers may finish normally while this process
   # creates its context. Their disappearance introduces no unknown context.
   # Every addition still participates in strict singleton attribution.
   if len(added)==1:break
   if len(added)>1 or time.monotonic()>=end:raise RuntimeError('Cannot uniquely attribute CUDA context across PID namespaces')
   time.sleep(.1)
  with lock(state/'registry.lock'):
   registry=read(registry_path);validate_registry(registry,int(os.environ['CTDG_SUPERVISOR_PID']))
   item=registry['workers'][str(os.getpid())];assert item['start_ticks']==birth(os.getpid())['start_ticks']
   item.update(host_gpu_pid=next(iter(added)),cuda_initialized=True,at=time.time())
   write(registry_path,registry)
  del marker
