"""CPU-only environment capture and fail-closed parity guard."""
import importlib
from pathlib import Path
import json
import os
import site
import sys


def capture():
    root = str(Path(__file__).resolve().parent.parent)
    return {'python': sys.version.split()[0], 'executable': sys.executable,
            'user_site_enabled': site.ENABLE_USER_SITE,
            'PYTHONNOUSERSITE': os.environ.get('PYTHONNOUSERSITE'),
            'sys_path': [str(p).replace(root, '${PACKAGE_ROOT}') for p in sys.path],
            'environment': {key: os.environ.get(key, '') for key in ['PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG']},
            'packages': {name: {'version': str(module.__version__), 'path': str(module.__file__)}
                         for name in ['numpy', 'pandas', 'torch', 'sklearn', 'scipy', 'tqdm']
                         for module in [importlib.import_module(name)]}}


def guard(expected_path, output):
    actual = capture()
    expected = json.loads(Path(expected_path).read_text())
    changes = {k: {'expected': expected.get(k), 'actual': actual.get(k)}
               for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'environment.actual.json').write_text(json.dumps(actual, indent=2) + '\n')
    (output / 'environment.diff.json').write_text(json.dumps(changes, indent=2) + '\n')
    if changes:
        raise RuntimeError('Environment differs: ' + ', '.join(changes))


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'source'))
    print(json.dumps(capture(), indent=2))
