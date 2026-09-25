"""Open-ended MAP-Elites: the niche set grows from what the search finds.

After each batch, images that no existing niche describes well found new
niches named by CoCa captions. An image founds a niche when:

1. **novel**: its CLIP image embedding is far from the text embeddings of
   every existing niche (``novelty_mode="text"``), from the image embeddings
   of every archived elite (``"image"``), or both (``"both"``);
2. **well described**: CoCa's caption for it, embedded by CoCa's own text
   encoder, matches the image (``cosine >= caption_threshold``); and
3. the caption is a usable, new name -- not empty, and not a duplicate of an
   existing niche's name (same content words in any order, or a CLIP text
   embedding within ``duplicate_threshold``) -- and the image is not already
   covered by a niche added earlier in the same batch.

The new niche's text embedding is the caption embedded by the scoring CLIP
model. Every current elite (and the batch that found the niche) then
competes for it, so it starts with the best image seen so far rather than
only the one that found it.

Scores must be comparable as niches are added, so this requires the
``cosine`` score mode: with ``softmax``, adding a niche would change every
other niche's scores. Only the 2D domain is supported for now.

``NicheDiscovery`` holds the policy and ``OpenEndedSearch`` the bookkeeping;
the underlying ``MapElites`` only needed a generic way to add niches.
"""
import re
from dataclasses import dataclass

import numpy as np
import torch

from .map_elites import MapElites


@dataclass
class Proposal:
    index: int            # position in the batch
    name: str             # cleaned caption
    novelty: float        # max cosine between the image and existing niche texts
    nearest: int          # index of that nearest niche (-1 if none)
    caption_score: float  # CoCa image/caption cosine
    text_feature: torch.Tensor = None
    image_novelty: float = float("nan")  # max cosine to archived elite images


NOVELTY_MODES = ("text", "image", "both")
_NAME_STOPWORDS = {"a", "an", "the", "and", "with", "of", "in", "on", "color", "colour"}


def name_key(name):
    """Order-insensitive content words: "a red and green color background" and
    "green and red background" share a key."""
    words = re.findall(r"[a-z]+", name.lower())
    return frozenset(w for w in words if w not in _NAME_STOPWORDS)


class NicheDiscovery:
    """Decides which images in a batch found new niches, and names them.

    ``novelty_mode`` picks what an image must be far from to count as novel:

    * ``text``: every niche's text embedding (max cosine < ``novelty_threshold``);
    * ``image``: every archived elite's image embedding (max cosine <
      ``image_novelty_threshold``) -- new-looking images, however well the
      current niche names already describe them;
    * ``both``: both of the above.

    A caption is rejected as a duplicate name when its content words match an
    existing niche's (ignoring order) or its text embedding is within
    ``duplicate_threshold`` cosine of one.
    """

    def __init__(self, captioner, novelty_threshold=0.26, caption_threshold=0.28,
                 max_per_batch=2, max_niches=None, min_words=2, novelty_mode="text",
                 image_novelty_threshold=0.89, duplicate_threshold=0.95):
        if novelty_mode not in NOVELTY_MODES:
            raise ValueError(f"unknown novelty mode {novelty_mode!r}")
        self.captioner = captioner
        self.novelty_threshold = novelty_threshold
        self.caption_threshold = caption_threshold
        self.max_per_batch = max_per_batch
        self.max_niches = max_niches
        self.min_words = min_words
        self.novelty_mode = novelty_mode
        self.image_novelty_threshold = image_novelty_threshold
        self.duplicate_threshold = duplicate_threshold
        self.stats = {"novel": 0, "captioned": 0, "low_caption_score": 0, "bad_name": 0,
                      "duplicate_name": 0, "covered_by_new": 0, "accepted": 0}

    @staticmethod
    def _max_sim(feats, others):
        if others is None or len(others) == 0:
            return np.full(len(feats), -np.inf), np.full(len(feats), -1)
        best, idx = (feats @ others.T).max(1)
        return best.cpu().numpy(), idx.cpu().numpy()

    def propose(self, images, feats, text_features, names, embed_texts, archive_features=None):
        """images (B,3,H,W); feats (B,D) normalised scorer image features;
        text_features (N,D) existing niche texts; embed_texts(list) -> (k,D);
        archive_features (M,D) image features of archived elites, excluding
        this batch (needed for the ``image``/``both`` modes)."""
        room = self.max_per_batch
        if self.max_niches is not None:
            room = min(room, self.max_niches - len(names))
        if room <= 0 or len(feats) == 0:
            return []
        use_text = self.novelty_mode in ("text", "both")
        use_image = self.novelty_mode in ("image", "both")
        novelty, nearest = self._max_sim(feats, text_features)
        image_novelty = np.full(len(feats), np.nan)
        novel = np.ones(len(feats), dtype=bool)
        if use_text:
            novel &= novelty < self.novelty_threshold
        if use_image:
            image_novelty, _ = self._max_sim(feats, archive_features)
            novel &= image_novelty < self.image_novelty_threshold
        cand = np.flatnonzero(novel)
        self.stats["novel"] += len(cand)
        if len(cand) == 0:
            return []
        # Caption the most novel few; captioning dominates the cost here.
        rank = image_novelty if use_image else novelty
        cand = cand[np.argsort(rank[cand])][:room * 2]
        imgs = images[cand]
        captions = self.captioner.caption(imgs)
        quality = self.captioner.similarity(imgs, captions)
        self.stats["captioned"] += len(cand)

        taken = {name_key(n) for n in names}
        accepted = []
        for i, caption, q in zip(cand, captions, quality):
            if len(accepted) >= room:
                break
            if q < self.caption_threshold:
                self.stats["low_caption_score"] += 1
                continue
            words = [w for w in caption.split() if any(c.isalpha() for c in w)]
            if len(words) < self.min_words:
                self.stats["bad_name"] += 1
                continue
            key = name_key(caption)
            text_feat = embed_texts([caption])[0]
            existing = [text_features] if len(text_features) else []
            existing += [p.text_feature[None] for p in accepted]
            dup_sim = float((torch.cat(existing) @ text_feat).max()) if existing else -1.0
            if key in taken or dup_sim >= self.duplicate_threshold:
                self.stats["duplicate_name"] += 1
                continue
            covered = (use_text and any(float(feats[i] @ p.text_feature) >= self.novelty_threshold
                                        for p in accepted)) or \
                      (use_image and any(float(feats[i] @ feats[p.index]) >= self.image_novelty_threshold
                                         for p in accepted))
            if covered:
                self.stats["covered_by_new"] += 1
                continue
            taken.add(key)
            accepted.append(Proposal(int(i), caption, float(novelty[i]), int(nearest[i]),
                                     float(q), text_feat, float(image_novelty[i])))
        self.stats["accepted"] += len(accepted)
        return accepted


class OpenEndedSearch:
    """MAP-Elites whose niches are added by a ``NicheDiscovery`` policy.

    Exposes the same surface the CLI uses for ``MapElites``: ``step``,
    ``scores``, ``elites``, ``evals``, ``state_dict``/``load_state_dict``.
    """

    def __init__(self, scorer, renderer, new_genome, mutate, discovery, seed_evals=500,
                 batch_size=32, curiosity=False, rng=None, on_new_niche=None):
        if scorer.mode != "cosine":
            raise ValueError("open-ended search needs --score cosine: softmax scores "
                             "change for every niche whenever a niche is added")
        if getattr(renderer, "num_views", 1) != 1:
            raise ValueError("open-ended search supports the 2D domain only for now")
        self.scorer, self.renderer, self.discovery = scorer, renderer, discovery
        self.on_new_niche = on_new_niche
        self.me = MapElites(scorer.num_niches, new_genome, mutate, self._evaluate,
                            seed_evals=seed_evals, batch_size=batch_size,
                            curiosity=curiosity, rng=rng)
        self.niche_info = [{"source": "seed"} for _ in range(scorer.num_niches)]
        self.elite_features = {}  # genome key -> scorer image feature (cpu)
        self._batch = None

    # -- the MapElites surface ----------------------------------------------

    @property
    def scores(self):
        return self.me.scores

    @property
    def elites(self):
        return self.me.elites

    @property
    def evals(self):
        return self.me.evals

    @property
    def niche_names(self):
        return self.scorer.niche_names

    def _evaluate(self, genomes):
        imgs = self.renderer.render_batch(genomes)
        feats = self.scorer.image_features(imgs)
        self._batch = (genomes, imgs, feats)
        return self.scorer.scores_from_features(feats)

    def step(self, max_evals=None):
        """One MAP-Elites batch, then niche discovery on that batch.
        Returns the list of niches added (dicts with name and provenance)."""
        before = self.me.evals
        self.me.step(max_evals)
        if self.me.evals == before:
            return []
        genomes, imgs, feats = self._batch
        self._batch = None
        self._remember_features(genomes, feats)
        if self.discovery is None:
            return []
        archive = None
        if self.discovery.novelty_mode != "text":
            batch_keys = {g.key for g in genomes}
            kept = [f for k, f in self.elite_features.items() if k not in batch_keys]
            archive = torch.stack(kept).to(feats.device) if kept else None
        proposals = self.discovery.propose(imgs, feats, self.scorer.text_features,
                                           self.scorer.niche_names, self.scorer.embed_texts,
                                           archive)
        added = [self._add_niche(p, genomes[p.index]) for p in proposals]
        if added:
            self._fill_new_niches(genomes, feats, [a["index"] for a in added])
            if self.on_new_niche:
                for a in added:
                    self.on_new_niche(a)
        return added

    # -- internals ---------------------------------------------------------

    def _add_niche(self, p, genome):
        (idx,) = self.scorer.add_niches([p.name])
        self.me.add_niches(1)
        nearest = self.scorer.niche_names[p.nearest] if p.nearest >= 0 else None
        info = {"source": "discovered", "eval": self.me.evals, "genome": genome.key,
                "caption_score": p.caption_score,
                "novelty": p.novelty if np.isfinite(p.novelty) else None, "nearest": nearest}
        if not np.isnan(p.image_novelty):  # image/both modes; None = nothing archived yet
            info["image_novelty"] = p.image_novelty if np.isfinite(p.image_novelty) else None
        self.niche_info.append(info)
        return {"index": idx, "name": p.name, **info}

    def _fill_new_niches(self, batch, batch_feats, new):
        pool = {g.key: (g, f) for g, f in zip(batch, batch_feats.cpu())}
        for g in self.me.elites:
            if g is not None and g.key in self.elite_features:
                pool.setdefault(g.key, (g, self.elite_features[g.key]))
        genomes = [g for g, _ in pool.values()]
        feats = torch.stack([f for _, f in pool.values()]).to(self.scorer.text_features.device)
        new = np.asarray(new)
        scores = (feats @ self.scorer.text_features[new].T).cpu().numpy().astype(np.float64)
        for g, s in zip(genomes, scores):
            self.me.offer(g, new, s)
        self._remember_features(batch, batch_feats)

    def _remember_features(self, genomes, feats):
        """Keep image features for current elites only (to score new niches)."""
        elite_keys = {g.key for g in self.me.elites if g is not None}
        for g, f in zip(genomes, feats.cpu()):
            if g.key in elite_keys:
                self.elite_features[g.key] = f
        for k in list(self.elite_features):
            if k not in elite_keys:
                del self.elite_features[k]

    # -- persistence -------------------------------------------------------

    def state_dict(self):
        return {**self.me.state_dict(), "open_ended": {
            "niche_info": self.niche_info,
            "elite_features": {k: v.numpy() for k, v in self.elite_features.items()},
            "discovery_stats": dict(self.discovery.stats) if self.discovery else {},
        }}

    def load_state_dict(self, state):
        """The scorer must already hold the same niches (seed + discovered)."""
        self.me.load_state_dict(state)
        oe = state["open_ended"]
        if len(oe["niche_info"]) != self.scorer.num_niches:
            raise ValueError("scorer niches do not match the checkpoint")
        self.niche_info = list(oe["niche_info"])
        self.elite_features = {k: torch.from_numpy(v) for k, v in oe["elite_features"].items()}
        if self.discovery and oe.get("discovery_stats"):
            self.discovery.stats.update(oe["discovery_stats"])


def discovered_names(niche_names, niche_info):
    """Names of the discovered niches, in the order they were added."""
    return [n for n, i in zip(niche_names, niche_info) if i.get("source") == "discovered"]

