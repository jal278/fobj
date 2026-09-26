"""Score images against niche text embeddings with CLIP (via open_clip).

Replaces the Caffe GoogLeNet classifier. Each niche is a text label; its
embedding is the normalised mean of the label run through each prompt
template. An image's score for a niche is either

* ``softmax`` (default): CLIP's zero-shot class probability over all niches,
  the analogue of the old classifier's softmax confidence -- so an elite at
  0.99 is an image CLIP is "sure" depicts that niche and nothing else; or
* ``cosine``: the raw image/text cosine similarity, which does not make
  niches compete with each other.

Any open_clip model works (``open_clip.list_pretrained()``), e.g. a bigger
``ViT-L-14``/``datacomp_xl_s13b_b90k`` or a SigLIP model.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "ViT-B-32-quickgelu"
DEFAULT_PRETRAINED = "openai"
DEFAULT_TEMPLATES = ("a photo of a {}.",)
CACHE_DIR = Path(os.environ.get("FOBJ_CACHE", Path.home() / ".cache" / "fobj"))


class ClipScorer:
    def __init__(self, niche_names, model=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED,
                 templates=DEFAULT_TEMPLATES, mode="softmax", device="cpu",
                 cache=True, dtype=torch.float32):
        import open_clip

        if mode not in ("softmax", "cosine"):
            raise ValueError(f"unknown score mode {mode!r}")
        self.mode = mode
        self.device = torch.device(device)
        self.model_name, self.pretrained = model, pretrained
        self.model, _, _ = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.dtype = dtype
        if dtype != torch.float32:
            self.model.to(dtype)
        self.tokenizer = open_clip.get_tokenizer(model)

        visual = self.model.visual
        size = getattr(visual, "image_size", 224)
        self.image_size = tuple(size) if isinstance(size, (tuple, list)) else (size, size)
        mean = getattr(visual, "image_mean", None) or open_clip.OPENAI_DATASET_MEAN
        std = getattr(visual, "image_std", None) or open_clip.OPENAI_DATASET_STD
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

        self.logit_scale = self.model.logit_scale.exp().item()
        bias = getattr(self.model, "logit_bias", None)
        self.logit_bias = bias.item() if bias is not None else 0.0

        self.niche_names = list(niche_names)
        self.templates = tuple(templates)
        self.text_features = self._text_features(cache)

    @property
    def num_niches(self):
        return len(self.niche_names)

    def _cache_path(self):
        h = hashlib.sha1()
        for part in (self.model_name, self.pretrained, str(self.dtype), *self.templates, "\0",
                     *self.niche_names):
            h.update(part.encode())
            h.update(b"\0")
        return CACHE_DIR / f"text-{h.hexdigest()[:16]}.pt"

    @torch.no_grad()
    def _text_features(self, cache):
        path = self._cache_path()
        if cache and path.exists():
            return torch.load(path, map_location=self.device)
        feats = self.embed_texts(self.niche_names, self.templates)
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feats.cpu(), path)
        return feats

    @torch.no_grad()
    def embed_texts(self, texts, templates=None):
        """Normalised text embeddings, (len(texts), dim).

        With ``templates``, each text is the normalised mean over the filled-in
        templates (prompt ensembling); without, the text is embedded as is.
        """
        templates = templates or ("{}",)
        feats = torch.zeros(len(texts), self._embed_dim(), device=self.device)
        for template in templates:
            prompts = [template.format(t) for t in texts]
            for i in range(0, len(prompts), 256):
                tokens = self.tokenizer(prompts[i:i + 256]).to(self.device)
                feats[i:i + 256] += F.normalize(self.model.encode_text(tokens).float(), dim=-1)
        return F.normalize(feats, dim=-1)

    def add_niches(self, names, templated=False):
        """Append niches; returns their indices. Discovered captions are
        usually full phrases, so by default they skip the prompt templates."""
        feats = self.embed_texts(names, self.templates if templated else None)
        start = len(self.niche_names)
        self.niche_names.extend(names)
        self.text_features = torch.cat([self.text_features, feats])
        return list(range(start, len(self.niche_names)))

    def scores_from_features(self, feats):
        """(B, dim) normalised image features -> (B, num_niches) numpy scores."""
        sims = feats @ self.text_features.T
        if self.mode == "cosine":
            out = sims
        else:
            out = (self.logit_scale * sims + self.logit_bias).softmax(dim=-1)
        return out.cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def _embed_dim(self):
        tokens = self.tokenizer(["x"]).to(self.device)
        return self.model.encode_text(tokens).shape[-1]

    @torch.no_grad()
    def image_features(self, images):
        """images: (B, 3, H, W) floats in [0, 1]. Returns normalised features."""
        images = images.to(self.device, torch.float32)
        if tuple(images.shape[-2:]) != self.image_size:
            # Antialiasing only matters when shrinking, and its kernel is missing
            # on some MPS builds; enlarging (e.g. 3D renders) uses plain bicubic.
            shrink = images.shape[-1] > self.image_size[1] or images.shape[-2] > self.image_size[0]
            images = F.interpolate(images, size=self.image_size, mode="bicubic",
                                   align_corners=False, antialias=shrink).clamp_(0, 1)
        images = ((images - self.mean) / self.std).to(self.dtype)
        return F.normalize(self.model.encode_image(images).float(), dim=-1)

    @torch.no_grad()
    def score(self, images, chunk=256):
        """Returns a (B, num_niches) numpy array of per-niche scores."""
        feats = torch.cat([self.image_features(images[i:i + chunk])
                           for i in range(0, len(images), chunk)])
        return self.scores_from_features(feats)


VIEW_AGGREGATIONS = ("geomean", "mean", "min", "prod")


def aggregate_views(scores, how="geomean"):
    """Combine (G, V, N) per-view scores into (G, N) per-object scores.

    ``geomean``/``prod`` suit probabilities (``softmax`` mode) and rank
    objects the same way the old code's product over views did; ``geomean``
    keeps the result on the same scale as a single view. With one view every
    option returns that view's scores unchanged.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[1] == 1:
        return scores[:, 0]
    if how == "mean":
        return scores.mean(1)
    if how == "min":
        return scores.min(1)
    if how in ("geomean", "prod"):
        if (scores < 0).any():
            raise ValueError(f"{how} view aggregation needs non-negative scores; "
                             "use mean or min with --score cosine")
        logs = np.log(np.maximum(scores, 1e-30))
        return np.exp(logs.mean(1) if how == "geomean" else logs.sum(1))
    raise ValueError(f"unknown view aggregation {how!r}")
