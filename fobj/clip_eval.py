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
                 cache=True):
        import open_clip

        if mode not in ("softmax", "cosine"):
            raise ValueError(f"unknown score mode {mode!r}")
        self.mode = mode
        self.device = torch.device(device)
        self.model_name, self.pretrained = model, pretrained
        self.model, _, _ = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device)
        self.model.eval()
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
        for part in (self.model_name, self.pretrained, *self.templates, "\0", *self.niche_names):
            h.update(part.encode())
            h.update(b"\0")
        return CACHE_DIR / f"text-{h.hexdigest()[:16]}.pt"

    @torch.no_grad()
    def _text_features(self, cache):
        path = self._cache_path()
        if cache and path.exists():
            return torch.load(path, map_location=self.device)
        feats = torch.zeros(len(self.niche_names), self._embed_dim(), device=self.device)
        for template in self.templates:
            prompts = [template.format(n) for n in self.niche_names]
            for i in range(0, len(prompts), 256):
                tokens = self.tokenizer(prompts[i:i + 256]).to(self.device)
                feats[i:i + 256] += F.normalize(self.model.encode_text(tokens).float(), dim=-1)
        feats = F.normalize(feats, dim=-1)
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feats.cpu(), path)
        return feats

    @torch.no_grad()
    def _embed_dim(self):
        tokens = self.tokenizer(["x"]).to(self.device)
        return self.model.encode_text(tokens).shape[-1]

    @torch.no_grad()
    def image_features(self, images):
        """images: (B, 3, H, W) floats in [0, 1]. Returns normalised features."""
        images = images.to(self.device, torch.float32)
        if tuple(images.shape[-2:]) != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode="bicubic",
                                   align_corners=False, antialias=True).clamp_(0, 1)
        images = (images - self.mean) / self.std
        return F.normalize(self.model.encode_image(images).float(), dim=-1)

    @torch.no_grad()
    def score(self, images):
        """Returns a (B, num_niches) numpy array of per-niche scores."""
        sims = self.image_features(images) @ self.text_features.T
        if self.mode == "cosine":
            out = sims
        else:
            out = (self.logit_scale * sims + self.logit_bias).softmax(dim=-1)
        return out.cpu().numpy().astype(np.float64)
