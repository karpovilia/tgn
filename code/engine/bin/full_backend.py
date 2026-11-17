"""Validation-only DyGFormer adapter over the frozen generic training backend."""
import hashlib
import random
import numpy as np
import torch
from backend import Backend
from data_runtime import load, get_neighbor_sampler
from exposure import InputTrace
from protocol import atomic, MiB


class FullBackend(Backend):
    def __init__(self, data, unit, cfg, package, permutations):
        super().__init__(data, unit, cfg)
        self.package = package
        self.permutations = permutations
        self.runtime_guard = lambda: None

    def pair(self, split, start, negative):
        self.runtime_guard()
        return super().pair(split, start, negative)

    def evaluate(self, split, path):
        if split != 'val':
            raise RuntimeError('Test evaluation is only reachable through frozen crossed evaluation')
        return super().evaluate(split, path)

    def selected_history_evaluations(self, out):
        original_unit = dict(self.unit, permutation_seed=None)
        original = load(self.package, original_unit, include_test=True)
        self.data = load(self.package, self.unit, include_test=True)
        assert np.array_equal(original['test'].edge_ids, self.data['test'].edge_ids)
        assert np.array_equal(original['negatives']['test'], self.data['negatives']['test'])
        histories = [None, *self.permutations] if self.unit['permutation_seed'] is None else [None, self.unit['permutation_seed']]
        saved_sampler = self.samplers['full']
        self.model.eval()
        before = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        reports = {}
        for perm in histories:
            data = original if perm is None else load(self.package, dict(original_unit, permutation_seed=perm), include_test=True)
            sampler = get_neighbor_sampler(data=data['full'], node_domain_size=len(data['node']), sample_neighbor_strategy='recent',
                                           time_scaling_factor=self.args['time_scaling_factor'], seed=1)
            self.samplers['full'] = sampler
            self.model[0].set_neighbor_sampler(sampler)
            name = 'native' if perm is None else 'perm' + str(perm)
            py_state = random.getstate(); np_state = np.random.get_state()
            cpu_rng = torch.get_rng_state().clone(); gpu_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
            positive, negative, traces = [], [], []
            q = original['test']
            with torch.inference_mode(), InputTrace(self.model[0]) as trace:
                for start in range(0, q.num_interactions, self.batch_size):
                    end = min(start + self.batch_size, q.num_interactions)
                    trace.begin(end - start)
                    pos, neg = self.pair('test', start, original['negatives']['test'][start:end])
                    positive.append(pos.cpu().numpy()); negative.append(neg.cpu().numpy())
                    traces.append(trace.finish())
            z = np.column_stack([np.concatenate(positive), np.concatenate(negative)])
            assert np.isfinite(z).all()
            assert py_state == random.getstate()
            after_np = np.random.get_state()
            assert np_state[0] == after_np[0] and np.array_equal(np_state[1], after_np[1]) and np_state[2:] == after_np[2:]
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert all(torch.equal(a, b) for a, b in zip(gpu_rng, torch.cuda.get_rng_state_all()))
            assert all(torch.equal(value, self.model.state_dict()[key].detach().cpu()) for key, value in before.items())
            payload = dict(query_id=q.edge_ids, src=q.src_node_ids, timestamp=q.node_interact_times,
                           candidate_id=np.column_stack([q.dst_node_ids, original['negatives']['test']]),
                           logit=z, split=np.array('test'), history=np.array(name),
                           **{key: np.concatenate([t[key] for t in traces]) for key in traces[0]})
            path = out / ('test_history_' + name + '.npz')
            atomic(path, lambda handle: np.savez_compressed(handle, **payload), 40 * MiB)
            reports[name] = {'queries': len(z), 'file': path.name,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.samplers['full'] = saved_sampler
        self.model[0].set_neighbor_sampler(saved_sampler)
        return reports
