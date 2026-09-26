"""Open-ended search with toy stand-ins for the renderer, CLIP and CoCa.

A "genome" is a 3-vector; its "image" is a constant-colour image of that
colour; CLIP features are the normalised colour; the captioner names an
image after its dominant channel and that name embeds to the matching axis.
"""
import copy
import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fobj.caption import clean_caption
from fobj.map_elites import MapElites
from fobj.open_ended import NicheDiscovery, OpenEndedSearch, name_key

AXES = {"reddish thing": 0, "greenish thing": 1, "bluish thing": 2}
_keys = itertools.count()


class G:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=np.float64)
        self.key = next(_keys)


class FakeRenderer:
    num_views = 1

    def render_batch(self, genomes):
        cols = torch.tensor(np.stack([np.clip(g.v, 0, 1) for g in genomes]), dtype=torch.float32)
        return cols[:, :, None, None].expand(-1, 3, 4, 4).clone()


class FakeScorer:
    mode = "cosine"

    def __init__(self, names=()):
        self.niche_names = list(names)
        self.text_features = self.embed_texts(self.niche_names)

    @property
    def num_niches(self):
        return len(self.niche_names)

    def embed_texts(self, texts, templates=None):
        out = torch.zeros(len(texts), 3)
        for i, t in enumerate(texts):
            # the axis whose colour word appears in the text, e.g. "thing reddish"
            out[i, next(v for k, v in AXES.items() if k.split()[0] in t)] = 1.0
        return out

    def image_features(self, imgs):
        return F.normalize(imgs.mean((2, 3)) + 1e-6, dim=-1)

    def add_niches(self, names, templated=False):
        start = len(self.niche_names)
        self.niche_names.extend(names)
        self.text_features = torch.cat([self.text_features, self.embed_texts(names)])
        return list(range(start, len(self.niche_names)))

    def scores_from_features(self, feats):
        return (feats @ self.text_features.T).numpy().astype(np.float64)


class FakeCaptioner:
    def __init__(self, quality=0.5):
        self.quality = quality
        self.calls = 0

    def caption(self, imgs):
        self.calls += len(imgs)
        names = list(AXES)
        return [names[int(c)] for c in imgs.mean((2, 3)).argmax(1)]

    def similarity(self, imgs, texts):
        return np.full(len(texts), self.quality)


def make_search(names=(), quality=0.5, novelty=0.9, **kw):
    rng = np.random.default_rng(0)
    scorer = FakeScorer(names)
    discovery = NicheDiscovery(FakeCaptioner(quality), novelty_threshold=novelty,
                               caption_threshold=0.3, **kw)
    return OpenEndedSearch(
        scorer, FakeRenderer(), lambda: G(rng.uniform(0, 1, 3)),
        lambda g: G(g.v + rng.normal(0, 0.1, 3)), discovery,
        seed_evals=8, batch_size=4, rng=rng)


def test_grows_niches_from_nothing():
    s = make_search()
    added = []
    while s.evals < 200:
        added += s.step(max_evals=200)
    assert sorted(s.niche_names) == sorted(AXES)
    assert [a["name"] for a in added] == s.niche_names
    assert all(i["source"] == "discovered" for i in s.niche_info)
    # every niche is held by the best image seen for it
    feats = s.scorer.image_features(s.renderer.render_batch(s.elites))
    np.testing.assert_allclose(np.diag(s.scorer.scores_from_features(feats)), s.scores, atol=1e-6)
    assert (s.scores > 0.9).all()


def test_new_niche_starts_with_best_existing_elite():
    s = make_search(names=["reddish thing"], novelty=0.5, max_per_batch=1)
    s.me.seed_evals = 0
    # A strongly green image sits in the archive as the (poor) red elite...
    green = G([0.2, 1.0, 0.0])
    s.me.insert(green, s.scorer.scores_from_features(
        s.scorer.image_features(s.renderer.render_batch([green])))[0])
    s._remember_features([green], s.scorer.image_features(s.renderer.render_batch([green])))
    # ...then a weakly green, novel image founds the "greenish thing" niche.
    s._evaluate([G([0.3, 0.7, 0.3])])  # stash a batch as step() would
    genomes, imgs, feats = s._batch
    props = s.discovery.propose(imgs, feats, s.scorer.text_features, s.scorer.niche_names,
                                s.scorer.embed_texts)
    assert [p.name for p in props] == ["greenish thing"]
    idx = s._add_niche(props[0], genomes[0])["index"]
    s._fill_new_niches(genomes, feats, [idx])
    assert s.elites[idx] is green  # the older, better green image wins the new niche


def test_rejections():
    # nothing is novel when the threshold is below every similarity
    s = make_search(names=list(AXES), novelty=0.0)
    while s.evals < 40:
        assert s.step() == []
    assert s.discovery.captioner.calls == 0
    # novel but badly described images never found niches
    s = make_search(quality=0.1)
    while s.evals < 40:
        assert s.step() == []
    assert s.discovery.stats["low_caption_score"] > 0
    # max_niches caps growth
    s = make_search(max_niches=1)
    while s.evals < 100:
        s.step()
    assert len(s.niche_names) == 1


def test_one_batch_does_not_add_the_same_idea_twice():
    s = make_search(max_per_batch=4)
    imgs = FakeRenderer().render_batch([G([1, 0, 0]), G([0.9, 0.1, 0]), G([0, 1, 0])])
    feats = s.scorer.image_features(imgs)
    props = s.discovery.propose(imgs, feats, s.scorer.text_features, [], s.scorer.embed_texts)
    assert [p.name for p in props] == ["reddish thing", "greenish thing"]


def test_state_roundtrip_and_softmax_rejected():
    s = make_search()
    while s.evals < 60:
        s.step()
    state = copy.deepcopy(s.state_dict())
    s2 = make_search(names=list(s.niche_names))
    s2.load_state_dict(state)
    assert s2.evals == s.evals and (s2.scores == s.scores).all()
    assert s2.niche_info == s.niche_info
    assert set(s2.elite_features) == {g.key for g in s.elites if g is not None}

    bad = FakeScorer()
    bad.mode = "softmax"
    with pytest.raises(ValueError):
        OpenEndedSearch(bad, FakeRenderer(), None, None, None)


def test_map_elites_add_niches_and_offer():
    me = MapElites(2, None, None, None)
    me.insert("a", np.array([0.5, 0.1]))
    new = me.add_niches(2)
    assert list(new) == [2, 3] and me.num_niches == 4 and me.elites[2:] == [None, None]
    held = me.offer("a", new, np.array([0.3, 0.2]))
    assert list(held) == [2, 3] and me.evals == 1
    assert list(me.offer("b", [3], [0.1])) == [] and me.elites[3] == "a"


def test_clean_caption():
    assert clean_caption("<start_of_text>free stock photo : a red , blue design . png"
                         "<end_of_text>junk") == "a red, blue design"
    assert clean_caption("royalty free stock illustration of a free kick") == "a free kick"
    assert clean_caption("illustration of a green and pink color background") == \
        "a green and pink color background"
    assert clean_caption("an image of a green and red light") == "a green and red light"
    assert clean_caption("image of the day") == "image of the day"
    assert clean_caption("stock vector illustration of a red ring") == "a red ring"
    assert clean_caption("<start_of_text>free stock photo : illustration of a blue and purple "
                         "background <end_of_text>") == "a blue and purple background"
    for junk in ["green and yellow light effect stock video footage",
                 "green and yellow light effect on pngtree",
                 "green and yellow light effect for cover book, poster o",
                 "green and yellow light effect no people day close"]:
        assert clean_caption(junk) == "green and yellow light effect", junk
    assert clean_caption("vector illustration of a green and yellow background") == \
        "a green and yellow background"


def test_duplicate_names_rejected():
    assert name_key("a red and green color background") == name_key("green and red background")
    assert name_key("red and green") != name_key("red and blue")
    # same words, different order
    s = make_search(names=["thing reddish"], novelty=1.1)
    imgs = FakeRenderer().render_batch([G([1, 0, 0])])
    props = s.discovery.propose(imgs, s.scorer.image_features(imgs), s.scorer.text_features,
                                s.scorer.niche_names, s.scorer.embed_texts)
    assert props == [] and s.discovery.stats["duplicate_name"] == 1
    # different words, but the text embedding is (here: exactly) the same
    s = make_search(names=["very reddish thing indeed"], novelty=1.1)
    props = s.discovery.propose(imgs, s.scorer.image_features(imgs), s.scorer.text_features,
                                s.scorer.niche_names, s.scorer.embed_texts)
    assert props == [] and s.discovery.stats["duplicate_name"] == 1
    # the embedding check is off above 1
    s = make_search(names=["very reddish thing indeed"], novelty=1.1, duplicate_threshold=1.01)
    props = s.discovery.propose(imgs, s.scorer.image_features(imgs), s.scorer.text_features,
                                s.scorer.niche_names, s.scorer.embed_texts)
    assert [p.name for p in props] == ["reddish thing"]


def test_image_novelty_mode():
    s = make_search(novelty_mode="image", image_novelty_threshold=0.9, max_per_batch=4)
    imgs = FakeRenderer().render_batch([G([1, 0.05, 0]), G([0, 1, 0])])
    feats = s.scorer.image_features(imgs)
    archive = s.scorer.image_features(FakeRenderer().render_batch([G([1, 0, 0])]))
    props = s.discovery.propose(imgs, feats, s.scorer.text_features, [], s.scorer.embed_texts,
                                archive)
    # the red image looks like the archived one; only the green one is novel
    assert [p.name for p in props] == ["greenish thing"]
    assert props[0].image_novelty < 0.9
    # "both" also needs text novelty: a green niche name already covers it
    s = make_search(names=["greenish thing"], novelty=0.5, novelty_mode="both",
                    image_novelty_threshold=0.9)
    assert s.discovery.propose(imgs, feats, s.scorer.text_features, s.scorer.niche_names,
                               s.scorer.embed_texts, archive) == []
    with pytest.raises(ValueError):
        NicheDiscovery(None, novelty_mode="nope")


def test_image_mode_search_grows_niches():
    s = make_search(novelty_mode="image", image_novelty_threshold=0.9)
    while s.evals < 300:
        s.step()
    assert sorted(s.niche_names) == sorted(AXES)
    assert all("image_novelty" in i for i in s.niche_info)
    assert s.niche_info[0]["image_novelty"] is None  # found before anything was archived
