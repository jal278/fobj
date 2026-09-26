"""Vendored, fixed-up subset of Uber's PyTorch-NEAT (Apache-2.0).

Upstream: https://github.com/uber-research/PyTorch-NEAT
(commit dee5f0adb22faf7d563c75ce28a8e3d6923d1b22). Only the CPPN pieces are
kept; see the module docstrings for what changed and ``LICENSE`` for terms.
"""
from .cppn import CPPN, create_cppn

__all__ = ["CPPN", "create_cppn"]
