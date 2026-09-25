"""Command line entry points: ``python -m fobj run`` and ``python -m fobj export``."""
import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import clip_eval
from .genome import DEFAULT_CONFIG, GenomeFactory, load_config
from .map_elites import MapElites
from .niches import load_niches
from .render import Renderer2D, to_pil

CHECKPOINT_VERSION = 1


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _write_config_copy(out, config_path):
    text = Path(config_path).read_text()
    (out / "cppn.cfg").write_text(text)
    return text


def _args_dict(args):
    return {k: v for k, v in vars(args).items() if k != "func"}


def save_checkpoint(path, me, args, niche_names, config_text):
    state = {
        "version": CHECKPOINT_VERSION,
        "map_elites": me.state_dict(),
        "niche_names": niche_names,
        "config_text": config_text,
        "args": _args_dict(args),
        "python_random": random.getstate(),
    }
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def config_from_text(text, out):
    path = Path(out) / "cppn.cfg"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return load_config(path)


def summarize(scores, thresholds=(0.1, 0.5, 0.9)):
    s = scores[np.isfinite(scores)]
    parts = [f"mean={s.mean():.4f}" if s.size else "mean=nan"]
    parts += [f">{t}:{int((s > t).sum())}" for t in thresholds]
    return " ".join(parts)


def contact_sheet(renderer, me, niche_names, path, top=64, cols=8, thumb=128):
    """Grid of the ``top`` best elites with their niche name and score."""
    from PIL import Image, ImageDraw

    order = [i for i in np.argsort(-me.scores) if me.elites[i] is not None][:top]
    if not order:
        return
    rows = (len(order) + cols - 1) // cols
    label_h = 14
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(order):
        img = to_pil(renderer.render(me.elites[i])).resize((thumb, thumb))
        x, y = (k % cols) * thumb, (k // cols) * (thumb + label_h)
        sheet.paste(img, (x, y))
        draw.text((x + 2, y + thumb + 1), f"{niche_names[i][:16]} {me.scores[i]:.2f}", fill="black")
    sheet.save(path)


def cmd_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    resume = None
    if args.resume:
        resume = load_checkpoint(args.resume)
        niche_names = resume["niche_names"]
        config = config_from_text(resume["config_text"], out)
        config_text = resume["config_text"]
    else:
        niche_names = load_niches(args.niches)
        config_text = _write_config_copy(out, args.config)
        config = load_config(out / "cppn.cfg")

    seed = args.seed if args.seed is not None else int(time.time())
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"device={device} niches={len(niche_names)} clip={args.clip_model}/{args.clip_pretrained}")
    scorer = clip_eval.ClipScorer(niche_names, model=args.clip_model,
                                  pretrained=args.clip_pretrained,
                                  templates=args.prompt or clip_eval.DEFAULT_TEMPLATES,
                                  mode=args.score, device=device)
    renderer = Renderer2D(config, size=args.size or scorer.image_size[0], device=device)

    def evaluate(genomes):
        return scorer.score(renderer.render_batch(genomes))

    existing = [g for g in resume["map_elites"]["elites"] if g is not None] if resume else ()
    factory = GenomeFactory(config, existing)
    me = MapElites(len(niche_names), factory.new, factory.mutate, evaluate,
                   seed_evals=args.seed_evals, batch_size=args.batch_size,
                   curiosity=args.curiosity, rng=np.random.default_rng(seed))
    if resume:
        me.load_state_dict(resume["map_elites"])
        random.setstate(resume["python_random"])
        print(f"resumed from {args.resume} at {me.evals} evals")

    (out / "run.json").write_text(json.dumps({**_args_dict(args), "seed": seed, "device": device}, indent=2))
    ckpt = out / "checkpoint.pkl"
    t0, e0 = time.time(), me.evals
    next_log = me.evals + args.log_every
    next_save = me.evals + args.save_every
    while me.evals < args.evals:
        me.step(max_evals=args.evals)
        if me.evals >= next_log or me.evals >= args.evals:
            rate = (me.evals - e0) / max(time.time() - t0, 1e-9)
            best = int(np.argmax(me.scores))
            print(f"evals={me.evals} {summarize(me.scores)} "
                  f"best={niche_names[best]!r}:{me.scores[best]:.3f} ({rate:.1f} evals/s)",
                  flush=True)
            next_log += args.log_every
        if me.evals >= next_save or me.evals >= args.evals:
            save_checkpoint(ckpt, me, args, niche_names, config_text)
            contact_sheet(renderer, me, niche_names, out / "top.png")
            next_save += args.save_every
    print(f"done: {ckpt}")


def cmd_export(args):
    state = load_checkpoint(args.checkpoint)
    out = Path(args.out or Path(args.checkpoint).parent / "elites")
    out.mkdir(parents=True, exist_ok=True)
    config = config_from_text(state["config_text"], out)
    renderer = Renderer2D(config, size=args.size, device=pick_device(args.device))
    names = state["niche_names"]
    me_state = state["map_elites"]
    scores, elites = np.asarray(me_state["scores"]), me_state["elites"]

    order = [i for i in np.argsort(-scores) if elites[i] is not None]
    if args.min_score is not None:
        order = [i for i in order if scores[i] >= args.min_score]
    if args.top:
        order = order[:args.top]
    rows = []
    for rank, i in enumerate(order):
        fname = f"{rank:04d}_{i:04d}_{names[i].replace('/', '_').replace(' ', '_')[:40]}.png"
        to_pil(renderer.render(elites[i])).save(out / fname)
        rows.append({"rank": rank, "niche": int(i), "name": names[i],
                     "score": float(scores[i]), "file": fname})
    (out / "elites.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} images to {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fobj", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run MAP-Elites with CLIP-defined niches")
    r.add_argument("--out", default="runs/default", help="output directory")
    r.add_argument("--evals", type=int, default=100_000, help="total evaluations")
    r.add_argument("--seed-evals", type=int, default=500, help="random genomes before mutating elites")
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--niches", default="imagenet",
                   help="'imagenet' (1000 WordNet classes) or a file with one niche name per line")
    r.add_argument("--config", default=str(DEFAULT_CONFIG), help="neat-python config file")
    r.add_argument("--clip-model", default=clip_eval.DEFAULT_MODEL)
    r.add_argument("--clip-pretrained", default=clip_eval.DEFAULT_PRETRAINED)
    r.add_argument("--prompt", action="append",
                   help="prompt template with {} for the niche name; repeat to ensemble "
                        f"(default: {clip_eval.DEFAULT_TEMPLATES[0]!r})")
    r.add_argument("--score", choices=["softmax", "cosine"], default="softmax")
    r.add_argument("--curiosity", action="store_true",
                   help="prefer parents from niches that are still improving (old --map_opt)")
    r.add_argument("--size", type=int, default=None, help="render size (default: CLIP input size)")
    r.add_argument("--device", default="auto")
    r.add_argument("--log-every", type=int, default=1000)
    r.add_argument("--save-every", type=int, default=10_000)
    r.add_argument("--resume", default=None, help="checkpoint.pkl to continue from")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("export", help="render the elites in a checkpoint to PNGs")
    e.add_argument("checkpoint")
    e.add_argument("--out", default=None, help="default: <checkpoint dir>/elites")
    e.add_argument("--size", type=int, default=512)
    e.add_argument("--top", type=int, default=None)
    e.add_argument("--min-score", type=float, default=None)
    e.add_argument("--device", default="auto")
    e.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
