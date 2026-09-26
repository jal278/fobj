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
"""Torch versions of neat-python's built-in activation functions.

Modified from PyTorch-NEAT: each function now matches neat-python's scalar
definition exactly (including input scaling and clamping). Upstream's ``sin``
was ``sin(x)`` where neat-python uses ``sin(5x)``, upstream ``gauss`` skipped
neat-python's clamp, and most of neat-python's activations were missing.
"""
import torch
import torch.nn.functional as F


def sigmoid_activation(x):
    return torch.sigmoid(torch.clamp(5.0 * x, -60.0, 60.0))


def tanh_activation(x):
    return torch.tanh(torch.clamp(2.5 * x, -60.0, 60.0))


def sin_activation(x):
    return torch.sin(torch.clamp(5.0 * x, -60.0, 60.0))


def gauss_activation(x):
    return torch.exp(-5.0 * torch.clamp(x, -3.4, 3.4) ** 2)


def relu_activation(x):
    return F.relu(x)


def elu_activation(x):
    return torch.where(x > 0, x, torch.expm1(torch.clamp(x, max=0.0)))


def lelu_activation(x):
    return torch.where(x > 0, x, 0.005 * x)


def selu_activation(x):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return torch.where(x > 0, lam * x, lam * alpha * torch.expm1(torch.clamp(x, max=0.0)))


def softplus_activation(x):
    return 0.2 * F.softplus(torch.clamp(5.0 * x, -60.0, 60.0))


def identity_activation(x):
    return x


def clamped_activation(x):
    return torch.clamp(x, -1.0, 1.0)


def inv_activation(x):
    safe = torch.where(x == 0, torch.ones_like(x), x)
    return torch.where(x == 0, torch.zeros_like(x), 1.0 / safe)


def log_activation(x):
    return torch.log(torch.clamp(x, min=1e-7))


def exp_activation(x):
    return torch.exp(torch.clamp(x, -60.0, 60.0))


def abs_activation(x):
    return torch.abs(x)


def hat_activation(x):
    return torch.clamp(1.0 - torch.abs(x), min=0.0)


def square_activation(x):
    return x**2


def cube_activation(x):
    return x**3


str_to_activation = {
    "sigmoid": sigmoid_activation,
    "tanh": tanh_activation,
    "sin": sin_activation,
    "gauss": gauss_activation,
    "relu": relu_activation,
    "elu": elu_activation,
    "lelu": lelu_activation,
    "selu": selu_activation,
    "softplus": softplus_activation,
    "identity": identity_activation,
    "clamped": clamped_activation,
    "inv": inv_activation,
    "log": log_activation,
    "exp": exp_activation,
    "abs": abs_activation,
    "hat": hat_activation,
    "square": square_activation,
    "cube": cube_activation,
}
