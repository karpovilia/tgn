"""Pure stdlib transactional protocol. No model imports, subprocesses or network."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

SCHEMA = "ctdg.full.v1"
GPU = "GPU-98a79194-9410-fe79-1c9e-796ab593b131"
MiB = 1024**2


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(MiB), b""): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic(path, writer, limit=64*MiB):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        writer(f)
        if f.tell() > limit: raise RuntimeError("artifact size limit exceeded: " + str(path))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path); fsync_dir(path.parent)


def write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    atomic(path, lambda f: f.write(raw), 8*MiB)


@contextlib.contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def contained(root, name):
    path = Path(root) / name
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("unsafe manifest path " + name)
    return path


def check_files(root, files):
    for name, meta in files.items():
        p = contained(root, name)
        if not p.is_file() or p.stat().st_size != meta["bytes"] or digest(p) != meta["sha256"]:
            raise RuntimeError("hash mismatch: " + str(p))


def manifest(root, exclude=()):
    return {str(p.relative_to(root)): {"sha256":digest(p),"bytes":p.stat().st_size}
            for p in sorted(Path(root).rglob("*")) if p.is_file() and str(p.relative_to(root)) not in exclude and not p.name.endswith(".tmp")}


def verify_package(package, expected_digest):
    if digest(Path(package)/"manifest.json") != expected_digest:
        raise RuntimeError("package manifest differs from launch contract")
    meta = read(Path(package)/"manifest.json")
    check_files(package, meta["files"])
    return meta


def early_update(early, ap, epoch, minimum=20, patience=10, maximum=100):
    if not math.isfinite(ap) or not 0 <= ap <= 1: raise ValueError("invalid validation AP")
    out = dict(early)
    improved = out.get("best_ap") is None or ap > out["best_ap"]
    if improved: out.update(best_ap=ap, best_epoch=epoch, bad_epochs=0)
    else: out["bad_epochs"] += 1
    # Strict greater retains earliest maximum, including ties.
    out["stop"] = epoch >= maximum or (epoch >= minimum and out["bad_epochs"] >= patience)
    return out, improved


class StateStore:
    """Atomic single-reference-set commit; immutable blobs, three retained roles."""
    def __init__(self, root, identity):
        self.root=Path(root); self.root.mkdir(parents=True, exist_ok=True); self.identity=identity
        ident=self.root/"identity.json"
        if ident.exists() and read(ident)!=identity: raise RuntimeError("foreign run state")
        if not ident.exists(): write(ident, identity)

    def refs(self):
        p=self.root/"refs.json"
        return read(p) if p.exists() else {}

    def save(self, state, serializer, best=False, budget10=False):
        # Fail before writing the new full checkpoint; never prune referenced blobs first.
        disk_guard(self.root, 1280*MiB, 80*MiB)
        path=self.root/"pending.pt"
        atomic(path, lambda f: serializer(state,f))
        sha=digest(path); dest=self.root/("state."+sha+".pt")
        if dest.exists(): path.unlink()
        else: os.replace(path,dest);fsync_dir(self.root)
        refs=self.refs();refs["latest"]={"path":dest.name,"sha256":sha,"bytes":dest.stat().st_size}
        if best: refs["best"]=refs["latest"]
        if budget10: refs["budget10"]=refs["latest"]
        write(self.root/"refs.json",refs)
        keep={x["path"] for x in refs.values()}
        for old in self.root.glob("state.*.pt"):
            if old.name not in keep: old.unlink()
        return refs

    def load(self, role, loader):
        meta=self.refs()[role];p=contained(self.root,meta["path"])
        check_files(self.root,{meta["path"]:meta})
        state=loader(p)
        if state["identity"]!=self.identity: raise RuntimeError("checkpoint identity mismatch")
        return state


def disk_guard(root, cap, reserve):
    root=Path(root)
    used=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if used+reserve>cap or shutil.disk_usage(root).free<reserve:
        raise OSError("bounded storage unavailable; preserve state, no new work")


def accept(root, identity, result, required):
    root=Path(root)
    for name in required:
        if not contained(root,name).is_file(): raise RuntimeError("required output missing: "+name)
    if result["status"]!="complete" or result["training_epochs"]<20 or result["training_epochs"]>100:
        raise RuntimeError("incomplete training")
    if result["test_queries"]!=result["expected_test_queries"] or result["validation_queries"]!=result["expected_validation_queries"]:
        raise RuntimeError("prediction coverage incomplete")
    write(root/"result.json",result)
    files={n:m for n,m in manifest(root,exclude=("accepted.json","heartbeat.json","worker.lock")).items() if not n.startswith("attempts/")}  # Driver logs finalize after worker acceptance; snapshot verifies them separately.
    payload={"schema":SCHEMA,"identity":identity,"status":"complete","files":files}
    write(root/"accepted.json",payload)
    return payload


def verified_acceptance(root, identity):
    p=Path(root)/"accepted.json"
    if not p.exists(): return False
    receipt=read(p)
    if receipt["identity"]!=identity or receipt["status"]!="complete": raise RuntimeError("wrong acceptance")
    check_files(root,receipt["files"])
    return True


def coverage(expected, accepted):
    e=set(expected); a=set(accepted)
    return {"status":"complete" if e==a else "incomplete","expected":len(e),"accepted":len(a),"missing":sorted(e-a),"unexpected":sorted(a-e)}


def mirror_tree(source, target):
    """Idempotent transfer to a root-attested durable filesystem, manifest last."""
    source=Path(source);target=Path(target);target.mkdir(parents=True,exist_ok=True)
    files=manifest(source,exclude=("worker.lock","heartbeat.json","snapshot.json"))
    for name,m in files.items():
        src=source/name;dst=contained(target,name)
        if dst.exists() and digest(dst)==m["sha256"]: continue
        atomic(dst,lambda f,s=src: _copy(s,f),max(m["bytes"],1))
    check_files(target,files)
    write(target/"snapshot.json",{"files":files,"saved_at":time.time()})
    # Old unreferenced state blobs are reproducible history, not accepted outputs.
    for old in target.glob("state.*.pt"):
        if old.name not in files: old.unlink()
    return digest(target/"snapshot.json")


def _copy(src, f):
    with Path(src).open("rb") as r: shutil.copyfileobj(r,f,MiB)


def restore_tree(source,target):
    source=Path(source); snap=read(source/"snapshot.json")
    check_files(source,snap["files"])
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    for name,m in snap["files"].items():
        dst=contained(target,name)
        if dst.exists() and digest(dst)==m["sha256"]: continue
        atomic(dst,lambda f,s=source/name:_copy(s,f),max(m["bytes"],1))
