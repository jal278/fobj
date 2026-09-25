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
"""Torch versions of neat-python's built-in aggregation functions.

Each takes a non-empty list of equally shaped tensors (the weighted inputs of
a node). Modified from PyTorch-NEAT, which only knew ``sum`` and ``prod`` --
and neat-python calls the latter ``product``, so upstream raised a KeyError
on any genome using it.
"""
import torch


def sum_aggregation(xs):
    return torch.stack(xs).sum(0)


def product_aggregation(xs):
    return torch.stack(xs).prod(0)


def max_aggregation(xs):
    return torch.stack(xs).max(0).values


def min_aggregation(xs):
    return torch.stack(xs).min(0).values


def maxabs_aggregation(xs):
    s = torch.stack(xs)
    idx = s.abs().argmax(0, keepdim=True)
    return s.gather(0, idx).squeeze(0)


def median_aggregation(xs):
    # neat-python averages the two middle values for even-length inputs.
    s = torch.stack(xs).sort(0).values
    n = s.shape[0]
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean_aggregation(xs):
    return torch.stack(xs).mean(0)


str_to_aggregation = {
    "sum": sum_aggregation,
    "product": product_aggregation,
    "prod": product_aggregation,  # upstream PyTorch-NEAT spelling
    "max": max_aggregation,
    "min": min_aggregation,
    "maxabs": maxabs_aggregation,
    "median": median_aggregation,
    "mean": mean_aggregation,
}
