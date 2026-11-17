"""Restartable validation-only training stages; test scoring is unreachable."""
import time
from protocol import early_update, write, digest, check_files


def run_segment(backend, store, cfg, guard=lambda: None):
    receipt = store.root / 'accepted_full.json'
    if receipt.exists():
        from protocol import read
        accepted = read(receipt)
        assert accepted['identity'] == store.identity
        check_files(store.root, accepted['files'])
        return 0
    if store.refs():
        state = store.load('latest', backend.load)
        backend.restore(state['backend'])
    else:
        state = dict(identity=store.identity, phase='train', epoch=1, next_batch=0,
                     completed_epochs=0, early=dict(best_ap=None, best_epoch=None,
                     bad_epochs=0, stop=False), history=[], committed_batches=0)

    def commit(best=False):
        state['backend'] = backend.capture()
        store.save(state, backend.save, best=best)
        write(store.root / 'heartbeat.json', {k: state[k] for k in
              ['phase', 'epoch', 'next_batch', 'completed_epochs'] } | {'at': time.time()})

    guard()
    if state['phase'] == 'train':
        if state['next_batch'] == 0:
            backend.begin_epoch(state['epoch'])
        for index in range(state['next_batch'], backend.n_batches):
            guard()
            backend.train_batch(index, state['epoch'])
            state['next_batch'] = index + 1
            state['committed_batches'] += 1
            if state['next_batch'] % cfg['checkpoint_every_batches'] == 0:
                commit()
        state['phase'] = 'val'; state['next_batch'] = 0
        commit(); return 75
    if state['phase'] == 'val':
        epoch = state['epoch']
        report = backend.evaluate('val', store.root / 'predictions' / f'val_epoch{epoch:03d}.npz')
        state['early'], improved = early_update(state['early'], report['pooled_ap'], epoch,
                                                cfg['min_epochs'], cfg['patience'], cfg['max_epochs'])
        state['history'].append({'epoch': epoch, 'validation': report})
        state['completed_epochs'] = epoch; state['epoch'] = epoch + 1
        state['phase'] = 'crossed_test' if state['early']['stop'] else 'train'
        commit(best=improved); return 75
    if state['phase'] == 'crossed_test':
        selected = store.load('best', backend.load)
        backend.restore(selected['backend'])
        outputs = backend.selected_history_evaluations(store.root / 'predictions')
        state['crossed_test'] = outputs
        state['phase'] = 'done'; commit()
    if state['phase'] == 'done':
        refs = store.refs()
        required = [refs[role]['path'] for role in ['latest', 'best']]
        required += ['predictions/' + v['file'] for v in state['crossed_test'].values()]
        if state.get('reused_POC'):
            required.append('reused_POC.json')
        else:
            required += [f'predictions/val_epoch{e:03d}.npz' for e in range(1, state['completed_epochs'] + 1)]
        result = {k: v for k, v in state.items() if k != 'backend'}
        result.update(status='complete', split='test', test_inference_performed=True)
        write(store.root / 'result.json', result)
        required.append('result.json')
        files = {name: {'sha256': digest(store.root / name), 'bytes': (store.root / name).stat().st_size}
                 for name in required}
        write(receipt, {'identity': store.identity, 'status': 'complete', 'files': files})
        return 0
    raise RuntimeError('Unexpected phase: ' + state['phase'])
