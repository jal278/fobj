"""Caption images with CoCa (open_clip) and check captions against images.

CoCa is a CLIP model with an extra text decoder, so one model can both
describe an image and score how well a piece of text matches it. The decoder
is driven here with a small greedy loop instead of ``CoCa.generate``, which
depends on HuggingFace ``transformers`` internals (``BeamSearchScorer``) that
transformers 5 removed.
"""
import re

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "coca_ViT-B-32"
# The LAION-only checkpoint, not the MS-COCO finetuned one: finetuning for
# captioning left the COCO model's contrastive head unable to tell a caption's
# own image from others (on CPPN images: matched vs mismatched cosine
# 0.068 vs 0.062), while this one separates them (0.32 vs 0.22). Its captions
# read like web alt-text, which ``clean_caption`` tidies.
DEFAULT_PRETRAINED = "laion2b_s13b_b90k"
SOT, EOT = 49406, 49407  # CLIP BPE start/end-of-text tokens


class CocaCaptioner:
    def __init__(self, model=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED, device="cpu",
                 max_len=30, min_len=3, repetition_penalty=1.3, prefix=""):
        """``prefix`` (e.g. "a photo of a") starts every caption, steering CoCa
        towards naming things; it is stripped from the returned captions."""
        import open_clip

        self.device = torch.device(device)
        self.model_name, self.pretrained = model, pretrained
        self.model, _, _ = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self._decode = open_clip.decode
        visual = self.model.visual
        size = getattr(visual, "image_size", 224)
        self.image_size = tuple(size) if isinstance(size, (tuple, list)) else (size, size)
        mean = getattr(visual, "image_mean", None) or open_clip.OPENAI_DATASET_MEAN
        std = getattr(visual, "image_std", None) or open_clip.OPENAI_DATASET_STD
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self.max_len, self.min_len = max_len, min_len
        self.repetition_penalty = repetition_penalty
        self.prefix = prefix.strip()
        self.prefix_tokens = self.tokenizer.encode(self.prefix) if self.prefix else []

    def _prep(self, images):
        images = images.to(self.device, torch.float32)
        if tuple(images.shape[-2:]) != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode="bicubic",
                                   align_corners=False, antialias=True).clamp_(0, 1)
        return (images - self.mean) / self.std

    @torch.no_grad()
    def caption(self, images):
        """images: (B, 3, H, W) in [0, 1] -> list of B caption strings."""
        x = self._prep(images)
        latent, embs = self.model._encode_image(x)
        start = torch.tensor([SOT] + self.prefix_tokens, device=self.device)
        out = start.expand(len(x), -1).clone()
        done = torch.zeros(len(x), dtype=torch.bool, device=self.device)
        for step in range(self.max_len):
            logits = self.model(x, out, image_latent=latent, image_embs=embs,
                                output_labels=False)["logits"][:, -1].float()
            if self.repetition_penalty != 1.0:
                prev = logits.gather(1, out)
                prev = torch.where(prev > 0, prev / self.repetition_penalty,
                                   prev * self.repetition_penalty)
                logits.scatter_(1, out, prev)
            if step < self.min_len:
                logits[:, EOT] = -torch.inf
            nxt = logits.argmax(-1)
            nxt[done] = EOT
            out = torch.cat([out, nxt[:, None]], 1)
            done |= nxt == EOT
            if done.all():
                break
        return [clean_caption(self._decode(row), self.prefix) for row in out.cpu()]

    @torch.no_grad()
    def similarity(self, images, texts):
        """Cosine similarity between each image and its caption, (B,) numpy."""
        img = F.normalize(self.model.encode_image(self._prep(images)).float(), dim=-1)
        txt = F.normalize(self.model.encode_text(self.tokenizer(texts).to(self.device)).float(), dim=-1)
        return (img * txt).sum(-1).cpu().numpy().astype(np.float64)


# Web alt-text boilerplate that LAION-trained captioners like to emit.
_BOILERPLATE = [
    r"\broyalty[- ]free\b",
    r"\bfree(?= stock\b)",
    r"^\s*free\b",
    r"\bstock (photo|image|picture|vector|illustration|footage|fot[oó])s?\b( of)?",
    r"\b(vector|stock) (image|photo|graphics?)\b",
    r"\b(for )?(a )?free download\b",
    r"\bimages? for your (iphone|android|phone|desktop|computer|mobile)\b.*$",
    r"\bfor (your )?(iphone|android|phone|desktop|mobile|ipad)\b.*$",
    r"\b(with an image )?for (free use|the color)\b",
    r"\b(hd|4k) wallpapers?\b",
    r"\bstock (video )?footage\b",
    r"\b(on )?pngtree\b",
    r"\bfor (your )?(design|cover|poster|banner|book|website)\b.*$",
    r"\bimages?,\s*pictures?\b",
    r"\bno people\b.*$",
    r"\.?\s*\b(png|jpe?g|gif|svg|webp)\b",
    r"^\s*(photo|image|picture|vector|illustration)s?\s*(of)?\s*:",
    r"^\s*((an?|free|stock|vector|cartoon|royalty)\s+)*"
    r"(photo|image|picture|vector|illustration|drawing)s? of\s+(?=an?\b)",
]
_DANGLING = {"a", "an", "the", "and", "or", "of", "for", "with", "in", "on", "to", "your", "by", "at"}


def clean_caption(text, prefix=""):
    """Tidy a decoded caption into a short niche name."""
    text = text.replace("<start_of_text>", "").split("<end_of_text>")[0].strip()
    if prefix and text.lower().startswith(prefix.lower()):
        text = text[len(prefix):]
    for pat in _BOILERPLATE:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)      # "a , b" -> "a, b"
    text = re.split(r"\s[-–|]\s|[.;!?](?:\s|$)", text)[0]  # first clause only
    text = re.sub(r"(?:\b\d+\b[\s,x-]*){2,}", " ", text)  # "1 2 3 5 4" / "1 2 8 0 x 4"
    text = re.sub(r"^[\s:;,.\-|]+", "", text)
    words = text.split()
    while words and words[-1].lower().strip(",") in _DANGLING:
        words.pop()
    return " ".join(words).strip(" .,:;-|")
