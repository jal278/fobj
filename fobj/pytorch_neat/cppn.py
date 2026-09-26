# Copyright (c) 2018 Uber Technologies, Inc.
# Modifications copyright (c) 2026 the fobj authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""Evaluate neat-python genomes as CPPNs over whole tensors at once.

Derived from PyTorch-NEAT's ``cppn.py``. Upstream built a recursive tree of
``Node`` objects; this version compiles the genome into a flat, topologically
ordered program using neat-python's own ``feed_forward_layers`` so that it
computes exactly what ``neat.nn.FeedForwardNetwork`` computes, just
vectorised over every pixel/voxel. Fixes relative to upstream:

* nodes with no incoming connections returned their raw bias instead of
  ``activation(bias)`` (neat-python applies the activation);
* constant tensors were created on the default device/dtype, breaking GPU use;
* deep networks could hit Python's recursion limit;
* errors inside a node were swallowed and re-raised without the cause;
* works with neat-python >= 1.0 (``feed_forward_layers`` now returns a tuple,
  aggregation ``product`` vs ``prod``).
"""
import torch
from neat.graphs import feed_forward_layers

from .activations import str_to_activation
from .aggregations import str_to_aggregation


class CPPN:
    """A compiled, tensor-valued view of a feed-forward neat-python genome.

    ``leaf_names`` name the genome's inputs (in ``input_keys`` order) and
    ``node_names`` its outputs. Call with one tensor per leaf name; all must
    share a shape. Returns a tensor of shape ``(num_outputs, *shape)``.
    """

    def __init__(self, genome, config, leaf_names, node_names=None,
                 output_activation=None):
        gc = config.genome_config
        if len(leaf_names) != len(gc.input_keys):
            raise ValueError(
                f"expected {len(gc.input_keys)} leaf names, got {len(leaf_names)}")
        self.leaf_names = list(leaf_names)
        self.node_names = list(node_names) if node_names is not None else None
        self.input_keys = list(gc.input_keys)
        self.output_keys = list(gc.output_keys)

        connections = [cg.key for cg in genome.connections.values() if cg.enabled]
        result = feed_forward_layers(self.input_keys, self.output_keys, connections)
        # neat-python < 1.0 returned just the layers.
        layers = result[0] if isinstance(result, tuple) else result
        required = set().union(*layers) if layers else set()
        sources = required | set(self.input_keys)

        self.program = []
        for layer in layers:
            for node in sorted(layer):
                ng = genome.nodes[node]
                links = [(i, genome.connections[(i, o)].weight)
                         for (i, o) in connections if o == node and i in sources]
                if node in self.output_keys and output_activation is not None:
                    act = output_activation
                else:
                    act = str_to_activation[ng.activation]
                agg_name = ng.aggregation
                self.program.append((node, act, agg_name, str_to_aggregation[agg_name],
                                     ng.bias, ng.response, links))

    def __call__(self, **inputs):
        missing = set(self.leaf_names) - set(inputs)
        if missing:
            raise ValueError(f"missing CPPN inputs: {sorted(missing)}")
        ref = inputs[self.leaf_names[0]]
        values = {}
        for key, name in zip(self.input_keys, self.leaf_names):
            x = inputs[name]
            if x.shape != ref.shape:
                raise ValueError(
                    f"input {name!r} has shape {tuple(x.shape)}, expected {tuple(ref.shape)}")
            values[key] = x

        for node, act, agg_name, agg, bias, response, links in self.program:
            try:
                if links:
                    s = agg([values[i] * w for i, w in links])
                else:
                    # neat-python: product([]) == 1, every other aggregation of [] == 0
                    s = torch.full_like(ref, 1.0 if agg_name in ("product", "prod") else 0.0)
                values[node] = act(bias + response * s)
            except Exception as e:
                raise RuntimeError(f"failed to activate CPPN node {node}") from e

        outs = [values[k] if k in values else self._unconnected(k, ref)
                for k in self.output_keys]
        return torch.stack(outs)

    def _unconnected(self, key, ref):
        # Output nodes that feed_forward_layers dropped (no path from anything)
        # keep neat-python's initial value of 0.0.
        return torch.zeros_like(ref)


def create_cppn(genome, config, leaf_names, node_names, output_activation=None):
    """PyTorch-NEAT compatible entry point.

    Returns one callable per output node; each takes the same ``**inputs`` as
    ``CPPN.__call__`` and returns that output's tensor. Prefer using ``CPPN``
    directly, which evaluates all outputs in a single pass.
    """
    cppn = CPPN(genome, config, leaf_names, node_names, output_activation)
    return [_OutputView(cppn, i, name) for i, name in enumerate(node_names)]


class _OutputView:
    def __init__(self, cppn, index, name):
        self.cppn, self.index, self.name = cppn, index, name

    def __call__(self, **inputs):
        return self.cppn(**inputs)[self.index]

    def __repr__(self):
        return f"CPPNOutput({self.name})"
