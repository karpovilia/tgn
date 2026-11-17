"""Package the E1/E2 prediction panels into the E1_E2_extension.zip layout.

The manuscript's numerical replay (code/reproduce_extension.py) reads a zip
with MANIFEST.json, E1/report.json, E2/report.json, E1/predictions/.../test.npz
and E2/predictions/.../test_history_*.npz. This script rebuilds that archive
from the unpacked predictions/ tree in this repository so the replay can be
run without re-training or GPU access.

Usage:
    python build_extension_zip.py [--out E1_E2_extension.zip]
"""
import argparse, hashlib, json, os, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # code/
ROOT = HERE.parent                              # repository root
PRED = ROOT / 'predictions'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=HERE / 'E1_E2_extension.zip')
    args = p.parse_args()

    files = {}
    for sub in ['E1', 'E2']:
        base = PRED / sub / 'predictions'
        for panel in sorted(base.iterdir()):
            if panel.is_dir():
                for f in sorted(panel.glob('*.npz')):
                    files[str(f.relative_to(PRED))] = f
    for name in ['E1/report.json', 'E2/report.json']:
        f = PRED / name
        assert f.exists(), f
        files[name] = f
    assert len([k for k in files if k.endswith('.npz')]) == 224, 'expected 224 prediction panels'

    manifest = {}
    for rel, f in sorted(files.items()):
        data = f.read_bytes()
        manifest[rel] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_STORED) as z:
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=2, sort_keys=True))
        for rel, f in sorted(files.items()):
            z.write(f, rel)
    print(f'Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, '
          f'{len(manifest)} entries, {len([k for k in manifest if k.endswith(".npz")])} panels)')


if __name__ == '__main__':
    main()