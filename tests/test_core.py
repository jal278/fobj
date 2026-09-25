import copy
import random
import tempfile
from pathlib import Path

import neat
import numpy as np
import pytest
import torch

from fobj.genome import DEFAULT_CONFIG, GenomeFactory, load_config
from fobj.map_elites import MapElites
from fobj.niches import load_niches
from fobj.pytorch_neat import CPPN, create_cppn
from fobj.pytorch_neat.activations import str_to_activation
from fobj.pytorch_neat.aggregations import str_to_aggregation
from fobj.render import Renderer2D

ALL_ACTIVATIONS = sorted(neat.activations.ActivationFunctionSet().functions)
ALL_AGGREGATIONS = sorted(neat.aggregations.AggregationFunctionSet().functions)


def config_with(**overrides):
    text = DEFAULT_CONFIG.read_text()
    for key, value in overrides.items():
        lines = text.splitlines()
        lines = [f"{key} = {value}" if l.split("=")[0].strip() == key else l for l in lines]
        text = "\n".join(lines) + "\n"
    tmp = Path(tempfile.mkdtemp()) / "cfg"
    tmp.write_text(text)
    return load_config(tmp)


def test_every_neat_activation_and_aggregation_is_ported():
    assert set(ALL_ACTIVATIONS) <= set(str_to_activation)
    assert set(ALL_AGGREGATIONS) <= set(str_to_aggregation)


@pytest.mark.parametrize("seed", range(5))
def test_cppn_matches_neat_feed_forward(seed):
    """The torch CPPN must compute exactly what neat-python's network does."""
    random.seed(seed)
    config = config_with(
        activation_options=" ".join(ALL_ACTIVATIONS),
        activation_mutate_rate=0.5,
        aggregation_options=" ".join(ALL_AGGREGATIONS),
        aggregation_mutate_rate=0.3,
        response_mutate_rate=0.3, response_mutate_power=0.5,
        node_add_prob=0.5, conn_add_prob=0.5, conn_delete_prob=0.1, node_delete_prob=0.05,
    )
    factory = GenomeFactory(config)
    genome = factory.new()
    for _ in range(40):
        genome = factory.mutate(genome)

    pts = torch.rand(3, 257, dtype=torch.float64) * 2 - 1
    cppn = CPPN(genome, config, ["x", "y", "d"], ["r", "g", "b"])
    got = cppn(x=pts[0], y=pts[1], d=pts[2]).numpy()

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    want = np.array([net.activate(list(map(float, pts[:, j]))) for j in range(pts.shape[1])]).T
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    views = create_cppn(genome, config, ["x", "y", "d"], ["r", "g", "b"])
    np.testing.assert_allclose(views[1](x=pts[0], y=pts[1], d=pts[2]).numpy(), want[1],
                               rtol=1e-6, atol=1e-6)


def test_bias_only_node_is_activated():
    # Upstream PyTorch-NEAT returned the raw bias for nodes without inputs.
    config = load_config()
    genome = GenomeFactory(config).new()
    genome.connections.clear()
    out = genome.nodes[0]
    out.bias, out.activation, out.response = 0.3, "sin", 1.0
    x = torch.zeros(4, dtype=torch.float64)
    got = CPPN(genome, config, ["x", "y", "d"], ["r", "g", "b"])(x=x, y=x, d=x)
    np.testing.assert_allclose(got[0].numpy(), np.sin(5 * 0.3))


def test_mutate_leaves_parent_untouched_and_ids_unique():
    random.seed(0)
    config = load_config()
    factory = GenomeFactory(config)
    parent = factory.new()
    before = copy.deepcopy(parent)
    kids = [factory.mutate(parent) for _ in range(50)]
    assert str(parent) == str(before)
    assert len({k.key for k in kids} | {parent.key}) == 51

    g = parent
    for _ in range(200):
        g = factory.mutate(g)
    # Resuming from saved genomes must not reuse their node ids.
    factory2 = GenomeFactory(load_config(), existing=[g])
    child = g
    for _ in range(200):
        child = factory2.mutate(child)


def test_renderer_range_and_shape():
    random.seed(1)
    config = load_config()
    factory = GenomeFactory(config)
    imgs = Renderer2D(config, size=32).render_batch([factory.new() for _ in range(3)])
    assert imgs.shape == (3, 3, 32, 32)
    assert imgs.min() >= 0 and imgs.max() <= 1


def test_imagenet_niches():
    names = load_niches("imagenet")
    assert len(names) == 1000
    assert len(set(names)) == 1000
    assert names[0] == "tench"


def _toy_problem(n_niches=5):
    """Genomes are floats; niche i rewards closeness to i."""
    targets = np.arange(n_niches, dtype=float)
    rng = np.random.default_rng(0)

    def evaluate(gs):
        return np.exp(-np.abs(np.array(gs)[:, None] - targets[None, :]))

    return (lambda: float(rng.uniform(-10, 10)),
            lambda g: g + float(rng.normal(0, 0.3)), evaluate)


@pytest.mark.parametrize("curiosity", [False, True])
def test_map_elites_improves_and_is_consistent(curiosity):
    new, mutate, evaluate = _toy_problem()
    me = MapElites(5, new, mutate, evaluate, seed_evals=10, batch_size=4,
                   curiosity=curiosity, rng=np.random.default_rng(0))
    me.step()
    first = me.scores.copy()
    history = [first]
    while me.evals < 400:
        me.step(max_evals=400)
        history.append(me.scores.copy())
    assert me.evals == 400
    assert all((b >= a).all() for a, b in zip(history, history[1:]))
    assert (me.scores > 0.9).all()
    np.testing.assert_allclose(evaluate(me.elites)[np.arange(5), np.arange(5)], me.scores)

    me2 = MapElites(5, new, mutate, evaluate, rng=np.random.default_rng(1))
    me2.load_state_dict(copy.deepcopy(me.state_dict()))
    assert me2.evals == me.evals and (me2.scores == me.scores).all()
