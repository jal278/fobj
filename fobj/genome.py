"""neat-python genome operations used by MAP-Elites (replaces MultiNEAT).

MAP-Elites only needs three things from NEAT: make a random genome, copy a
genome and mutate the copy. neat-python exposes these directly; this module
wraps them and takes care of the bookkeeping neat-python normally leaves to
its ``Population``/``Reproduction`` classes (innovation tracker, node ids).
"""
import copy
import itertools
from pathlib import Path

import neat
from neat.innovation import InnovationTracker

DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "cppn2d.cfg"


def load_config(path=DEFAULT_CONFIG):
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))


class GenomeFactory:
    """Creates and mutates genomes that share one neat-python config.

    ``existing`` (e.g. genomes loaded from a checkpoint) seeds the genome,
    node and innovation counters so new ids never collide with old ones.
    """

    def __init__(self, config, existing=()):
        self.config = config
        gc = config.genome_config
        existing = list(existing)
        next_key = max((g.key for g in existing), default=-1) + 1
        next_node = max((k for g in existing for k in g.nodes), default=gc.num_outputs - 1) + 1
        last_innov = max((c.innovation for g in existing for c in g.connections.values()
                          if getattr(c, "innovation", None) is not None), default=0)
        self._keys = itertools.count(next_key)
        # neat-python lazily creates node_indexer from the first genome it sees;
        # make it global over everything we already have.
        gc.node_indexer = itertools.count(max(next_node, gc.num_outputs))
        gc.innovation_tracker = InnovationTracker(start_number=last_innov)

    def new(self):
        g = self.config.genome_type(next(self._keys))
        g.configure_new(self.config.genome_config)
        return g

    def mutate(self, parent):
        child = copy.deepcopy(parent)
        child.key = next(self._keys)
        child.fitness = None
        # Innovation numbers only matter for crossover, which MAP-Elites never
        # does; clearing the per-"generation" dedup table keeps it from growing
        # without bound over millions of evaluations.
        self.config.genome_config.innovation_tracker.reset_generation()
        child.mutate(self.config.genome_config)
        return child
