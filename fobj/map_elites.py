"""MAP-Elites over a fixed set of niches, one score per niche per candidate.

A port of the algorithm in ``legacy/melites.py``: every evaluated genome
produces a score for every niche and becomes the elite of each niche where it
beats the incumbent. After ``seed_evals`` random genomes, children are made by
mutating the elite of a randomly chosen niche. Candidates are generated and
scored in batches so the image model runs on full batches.

The niche set can grow during a run (``add_niches`` + ``offer``), which is
what ``fobj.open_ended`` builds on; plain runs never use it.

``curiosity=True`` is the old ``--map_opt`` mode: each niche has a budget of
``reset_tries`` selections that is spent when it is picked as a parent and
refilled whenever it (or a child of it) improves by more than
``improve_ratio``, so search concentrates on niches that are still improving.
"""
import numpy as np


class MapElites:
    def __init__(self, num_niches, new_genome, mutate, evaluate, seed_evals=500,
                 batch_size=32, curiosity=False, reset_tries=10, improve_ratio=1.05,
                 rng=None):
        """
        new_genome(): -> genome
        mutate(genome): -> new child genome (must not modify its argument)
        evaluate(list of genomes): -> (len, num_niches) array of scores
        """
        self.num_niches = num_niches
        self.new_genome = new_genome
        self.mutate = mutate
        self.evaluate = evaluate
        self.seed_evals = seed_evals
        self.batch_size = batch_size
        self.curiosity = curiosity
        self.reset_tries = reset_tries
        self.improve_ratio = improve_ratio
        self.rng = rng if rng is not None else np.random.default_rng()

        self.scores = np.full(num_niches, -np.inf)
        self.elites = [None] * num_niches
        # For each niche: (eval index when found, parent niche or -1)
        self.found_at = np.full(num_niches, -1, dtype=np.int64)
        self.parent_niche = np.full(num_niches, -1, dtype=np.int64)
        self.tries = np.full(num_niches, float(reset_tries))
        self.evals = 0
        self.last_children, self.last_parents = [], np.empty(0, dtype=np.int64)

    def _pick_parents(self, n):
        filled = np.flatnonzero(self.scores > -np.inf)
        if not self.curiosity:
            return self.rng.choice(filled, size=n)
        parents = []
        for _ in range(n):
            w = self.tries[filled]
            if w.sum() <= 0:
                self.tries[:] = self.reset_tries
                w = self.tries[filled]
            niche = self.rng.choice(filled, p=w / w.sum())
            self.tries[niche] -= 1
            parents.append(niche)
        return np.array(parents)

    def step(self, max_evals=None):
        """Generate, evaluate and insert one batch. Returns the batch's scores.

        ``max_evals`` caps the batch so ``self.evals`` never exceeds it.
        """
        n = self.batch_size
        if max_evals is not None:
            n = min(n, max_evals - self.evals)
            if n <= 0:
                return np.empty((0, self.num_niches))
        if self.evals < self.seed_evals or not np.isfinite(self.scores).any():
            # Random genomes while seeding, or while no niche has an elite yet
            # (possible when the niche set starts empty and grows).
            if self.evals < self.seed_evals:
                n = min(n, self.seed_evals - self.evals)
            parents = np.full(n, -1)
            children = [self.new_genome() for _ in range(n)]
        else:
            parents = self._pick_parents(n)
            children = [self.mutate(self.elites[p]) for p in parents]

        scores = np.asarray(self.evaluate(children), dtype=np.float64)
        if scores.shape != (len(children), self.num_niches):
            raise ValueError(f"evaluate returned shape {scores.shape}, expected "
                             f"{(len(children), self.num_niches)}")
        for child, parent, s in zip(children, parents, scores):
            self.insert(child, s, parent)
        self.last_children, self.last_parents = children, parents
        return scores

    def insert(self, genome, scores, parent=-1):
        """Insert a newly evaluated genome; ``scores`` covers every niche."""
        idx = self.offer(genome, np.arange(self.num_niches), scores, parent)
        self.evals += 1
        return idx

    def offer(self, genome, niches, scores, parent=-1):
        """Let an already evaluated genome compete for some niches only.

        ``scores[k]`` is its score for niche ``niches[k]``. Does not count as an
        evaluation. Returns the indices of the niches it now holds.
        """
        niches = np.asarray(niches, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        old = self.scores[niches]
        better = scores > old
        if self.curiosity:
            improved = better & (scores > self.improve_ratio * old)
            self.tries[niches[improved]] = self.reset_tries
            if parent >= 0 and improved.any():
                self.tries[parent] = self.reset_tries
        idx = niches[better]
        self.scores[idx] = scores[better]
        for i in idx:
            self.elites[i] = genome
        self.found_at[idx] = self.evals
        self.parent_niche[idx] = parent
        return idx

    def add_niches(self, n):
        """Append ``n`` empty niches; returns their indices."""
        start = self.num_niches
        self.num_niches += n
        self.scores = np.concatenate([self.scores, np.full(n, -np.inf)])
        self.elites.extend([None] * n)
        self.found_at = np.concatenate([self.found_at, np.full(n, -1, dtype=np.int64)])
        self.parent_niche = np.concatenate([self.parent_niche, np.full(n, -1, dtype=np.int64)])
        self.tries = np.concatenate([self.tries, np.full(n, float(self.reset_tries))])
        return np.arange(start, self.num_niches)

    # -- persistence -------------------------------------------------------

    def state_dict(self):
        return {
            "scores": self.scores, "elites": self.elites, "found_at": self.found_at,
            "parent_niche": self.parent_niche, "tries": self.tries, "evals": self.evals,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        self.scores = np.asarray(state["scores"], dtype=np.float64)
        self.num_niches = len(self.scores)
        self.elites = list(state["elites"])
        self.found_at = np.asarray(state["found_at"])
        self.parent_niche = np.asarray(state["parent_niche"])
        self.tries = np.asarray(state["tries"], dtype=np.float64)
        self.evals = int(state["evals"])
        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]
