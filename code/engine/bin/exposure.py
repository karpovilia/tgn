"""Trace actual DyGFormer inputs during inference, including batch-dependent padding.

No predictions, target labels or learned scores enter the exposure definition.
Hashes are only compared across histories at fixed model weights and batches.
"""
import hashlib
import numpy as np


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class InputTrace:
    def __init__(self, backbone):
        self.backbone = backbone
        self.original_features = backbone.get_features
        self.original_counts = backbone.neighbor_co_occurrence_encoder.count_nodes_appearances
        self.n = None
        self.handles = []

    def add(self, kind, values):
        for item in values:
            a = array(item)
            assert len(a) == self.n
            for i, row in enumerate(a):
                h = self.inputs[i] if kind == 'input' else self.transformer[i]
                h.update(str((row.shape, row.dtype.str)).encode())
                h.update(np.ascontiguousarray(row).tobytes())

    def begin(self, n):
        self.n = n
        self.inputs = [hashlib.sha256() for _ in range(n)]
        self.transformer = [hashlib.sha256() for _ in range(n)]
        self.feature_calls = self.count_calls = self.transformer_calls = 0

    def __enter__(self):
        def features(*args, **kwargs):
            out = self.original_features(*args, **kwargs)
            self.add('input', out)
            # The padding pattern is part of the actual sequence construction.
            self.add('input', [kwargs['padded_nodes_neighbor_ids'] == 0])
            self.feature_calls += 1
            return out

        def counts(*args, **kwargs):
            out = self.original_counts(*args, **kwargs)
            self.add('input', out)
            self.count_calls += 1
            return out

        def transformer(module, args):
            self.add('transformer', [args[0]])
            self.transformer_calls += 1

        self.backbone.get_features = features
        self.backbone.neighbor_co_occurrence_encoder.count_nodes_appearances = counts
        self.handles.append(self.backbone.transformers[0].register_forward_pre_hook(transformer))
        return self

    def finish(self):
        assert (self.feature_calls, self.count_calls, self.transformer_calls) == (4, 2, 2)
        return {
            'actual_input_sha256': np.array([h.hexdigest() for h in self.inputs], dtype='U64'),
            'transformer_input_sha256': np.array([h.hexdigest() for h in self.transformer], dtype='U64'),
        }

    def __exit__(self, *exc):
        self.backbone.get_features = self.original_features
        self.backbone.neighbor_co_occurrence_encoder.count_nodes_appearances = self.original_counts
        for handle in self.handles:
            handle.remove()
