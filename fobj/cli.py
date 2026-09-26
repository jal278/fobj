"""Command line entry points: ``python -m fobj run`` and ``python -m fobj export``."""
import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import caption, clip_eval
from .device import PRECISIONS, model_dtype, pick_device
from .genome import GenomeFactory, load_config
from .map_elites import MapElites
from .niches import load_niches
from .render import DEFAULT_CONFIGS, DEFAULT_SIZES, make_renderer, to_pil

CHECKPOINT_VERSION = 2
DATA = Path(__file__).resolve().parent / "data"


def _write_config_copy(out, config_path):
    text = Path(config_path).read_text()
    (out / "cppn.cfg").write_text(text)
    return text


def _args_dict(args):
    return {k: v for k, v in vars(args).items() if k != "func"}


def save_checkpoint(path, me, args, niche_names, config_text, render, view_agg,
                    score, open_ended=None):
    state = {
        "version": CHECKPOINT_VERSION,
        "map_elites": me.state_dict(),
        "niche_names": niche_names,
        "config_text": config_text,
        "render": render,
        "view_agg": view_agg,
        "score": score,
        "open_ended": open_ended,
        "args": _args_dict(args),
        "python_random": random.getstate(),
    }
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    # Version 1 checkpoints were always 2D, single view.
    state.setdefault("render", {"domain": "2d", "size": state["args"].get("size") or 224})
    state.setdefault("view_agg", "mean")
    state.setdefault("score", state["args"].get("score") or "softmax")
    state.setdefault("open_ended", None)
    return state


def image_strip(views):
    """(V, 3, H, W) -> (3, H, V * W), views side by side."""
    return torch.cat(list(views), dim=2)


def config_from_text(text, out):
    path = Path(out) / "cppn.cfg"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return load_config(path)


def summarize(scores, mode="softmax"):
    thresholds = (0.1, 0.5, 0.9) if mode == "softmax" else (0.25, 0.3, 0.35)
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
        # The domain, render settings and view aggregation come from the
        # checkpoint: the archive's scores are only comparable under them.
        resume = load_checkpoint(args.resume)
        niche_names = resume["niche_names"]
        config = config_from_text(resume["config_text"], out)
        config_text = resume["config_text"]
        render, view_agg = resume["render"], resume["view_agg"]
        score, open_ended = resume["score"], resume["open_ended"]
    else:
        niche_names = load_niches(args.niches)
        config_text = _write_config_copy(out, args.config or DATA / DEFAULT_CONFIGS[args.domain])
        config = load_config(out / "cppn.cfg")
        render = {"domain": args.domain, "size": args.size or DEFAULT_SIZES[args.domain]}
        if args.domain == "3d":
            render.update(voxels=args.voxels, views=args.views, fixed_bg=args.fixed_bg,
                          lighting=not args.no_lighting, march_step=args.march_step)
        score = args.score or ("cosine" if args.open_ended else "softmax")
        view_agg = args.view_agg or ("geomean" if score == "softmax" else "mean")
        open_ended = None
        if args.open_ended:
            if args.domain != "2d":
                raise SystemExit("--open-ended supports --domain 2d only for now")
            if score != "cosine":
                raise SystemExit("--open-ended needs --score cosine (softmax scores shift "
                                 "whenever a niche is added)")
            open_ended = {"novelty_mode": args.novelty_mode,
                          "novelty_threshold": args.novelty_threshold,
                          "image_novelty_threshold": args.image_novelty_threshold,
                          "duplicate_threshold": args.duplicate_threshold,
                          "caption_threshold": args.caption_threshold,
                          "max_per_batch": args.max_new_niches_per_batch,
                          "max_niches": args.max_niches,
                          "caption_model": args.caption_model or caption.DEFAULT_MODEL,
                          "caption_pretrained": args.caption_pretrained or caption.DEFAULT_PRETRAINED,
                          "caption_prefix": args.caption_prefix}

    seed = args.seed if args.seed is not None else int(time.time())
    random.seed(seed)
    torch.manual_seed(seed)

    discovered = []
    if resume and open_ended:
        info = resume["map_elites"]["open_ended"]["niche_info"]
        discovered = [n for n, i in zip(niche_names, info) if i["source"] == "discovered"]
        niche_names = [n for n, i in zip(niche_names, info) if i["source"] != "discovered"]
    print(f"device={device} precision={args.precision} niches={len(niche_names) + len(discovered)} "
          f"clip={args.clip_model}/{args.clip_pretrained} score={score}")
    scorer = clip_eval.ClipScorer(niche_names, model=args.clip_model,
                                  pretrained=args.clip_pretrained,
                                  templates=args.prompt or clip_eval.DEFAULT_TEMPLATES,
                                  mode=score, device=device,
                                  dtype=model_dtype(args.precision, device))
    if discovered:
        scorer.add_niches(discovered)
    niche_names = scorer.niche_names  # grows in open-ended runs
    render["size"] = render["size"] or scorer.image_size[0]
    renderer = make_renderer(render, config, device=device)
    print(f"render={render} view_agg={view_agg}")

    def evaluate(genomes):
        views = torch.cat([renderer.render_views(g) for g in genomes])
        scores = scorer.score(views).reshape(len(genomes), renderer.num_views, -1)
        return clip_eval.aggregate_views(scores, view_agg)

    existing = [g for g in resume["map_elites"]["elites"] if g is not None] if resume else ()
    factory = GenomeFactory(config, existing)
    if open_ended:
        me = make_open_ended(open_ended, scorer, renderer, factory, args, seed, device, out)
    else:
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
            extra = f" niches={len(niche_names)}" if open_ended else ""
            print(f"evals={me.evals} {summarize(me.scores, score)}{extra} "
                  f"best={niche_names[best]!r}:{me.scores[best]:.3f} ({rate:.1f} evals/s)",
                  flush=True)
            next_log += args.log_every
        if me.evals >= next_save or me.evals >= args.evals:
            save_checkpoint(ckpt, me, args, niche_names, config_text, render, view_agg,
                            score, open_ended)
            if open_ended:
                write_niches(out / "niches.json", niche_names, me.niche_info, me.scores,
                             me.discovery.stats)
            contact_sheet(renderer, me, niche_names, out / "top.png")
            next_save += args.save_every
    print(f"done: {ckpt}")


def make_open_ended(settings, scorer, renderer, factory, args, seed, device, out):
    from .caption import CocaCaptioner
    from .open_ended import NicheDiscovery, OpenEndedSearch

    print(f"open-ended: {settings}")
    captioner = CocaCaptioner(settings["caption_model"], settings["caption_pretrained"],
                              device=device, prefix=settings.get("caption_prefix", ""),
                              dtype=model_dtype(args.precision, device))
    discovery = NicheDiscovery(captioner, novelty_threshold=settings["novelty_threshold"],
                               caption_threshold=settings["caption_threshold"],
                               max_per_batch=settings["max_per_batch"],
                               max_niches=settings["max_niches"],
                               novelty_mode=settings.get("novelty_mode", "text"),
                               image_novelty_threshold=settings.get("image_novelty_threshold", 0.89),
                               duplicate_threshold=settings.get("duplicate_threshold", 1.01))
    log = open(out / "discoveries.jsonl", "a")

    def on_new_niche(info):
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        img = f", image novelty {fmt(info['image_novelty'])}" if "image_novelty" in info else ""
        print(f"  + niche {info['index']}: {info['name']!r} (caption {info['caption_score']:.3f}, "
              f"nearest text {info['nearest']!r} at {fmt(info['novelty'])}{img}, "
              f"eval {info['eval']})", flush=True)
        log.write(json.dumps(info) + "\n")
        log.flush()

    return OpenEndedSearch(scorer, renderer, factory.new, factory.mutate, discovery,
                           seed_evals=args.seed_evals, batch_size=args.batch_size,
                           curiosity=args.curiosity, rng=np.random.default_rng(seed),
                           on_new_niche=on_new_niche)


def write_niches(path, names, info, scores, stats):
    rows = [{"index": i, "name": n, "score": float(s) if np.isfinite(s) else None, **inf}
            for i, (n, inf, s) in enumerate(zip(names, info, scores))]
    path.write_text(json.dumps({"discovery_stats": stats, "niches": rows}, indent=1))


def cmd_export(args):
    state = load_checkpoint(args.checkpoint)
    out = Path(args.out or Path(args.checkpoint).parent / "elites")
    out.mkdir(parents=True, exist_ok=True)
    config = config_from_text(state["config_text"], out)
    renderer = make_renderer(state["render"], config, device=pick_device(args.device),
                             size=args.size)
    mesh = args.mesh and state["render"].get("domain") == "3d"
    names = state["niche_names"]
    me_state = state["map_elites"]
    scores, elites = np.asarray(me_state["scores"]), me_state["elites"]
    niche_info = me_state.get("open_ended", {}).get("niche_info")
    if args.discovered_only and niche_info:
        keep = {i for i, inf in enumerate(niche_info) if inf.get("source") == "discovered"}
    else:
        keep = None

    order = [i for i in np.argsort(-scores) if elites[i] is not None
             and (keep is None or i in keep)]
    if args.min_score is not None:
        order = [i for i in order if scores[i] >= args.min_score]
    if args.top:
        order = order[:args.top]
    rows = []
    for rank, i in enumerate(order):
        fname = f"{rank:04d}_{i:04d}_{names[i].replace('/', '_').replace(' ', '_')[:40]}.png"
        to_pil(image_strip(renderer.render_views(elites[i]))).save(out / fname)
        row = {"rank": rank, "niche": int(i), "name": names[i],
               "score": float(scores[i]), "file": fname}
        if niche_info:
            row.update({k: v for k, v in niche_info[i].items() if k != "genome"})
        if mesh:
            from .render3d import marching_cubes_mesh, save_ply

            m = marching_cubes_mesh(renderer, elites[i])
            if m is not None:
                row["mesh"] = fname[:-4] + ".ply"
                save_ply(out / row["mesh"], *m)
        rows.append(row)
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
    r.add_argument("--domain", choices=["2d", "3d"], default="2d",
                   help="2d: CPPN images; 3d: CPPN voxel objects rendered from several views")
    r.add_argument("--config", default=None,
                   help="neat-python config file (default: fobj/data/cppn<domain>.cfg)")
    r.add_argument("--clip-model", default=clip_eval.DEFAULT_MODEL)
    r.add_argument("--clip-pretrained", default=clip_eval.DEFAULT_PRETRAINED)
    r.add_argument("--prompt", action="append",
                   help="prompt template with {} for the niche name; repeat to ensemble "
                        f"(default: {clip_eval.DEFAULT_TEMPLATES[0]!r})")
    r.add_argument("--score", choices=["softmax", "cosine"], default=None,
                   help="softmax (default) or cosine (default and required with --open-ended)")
    r.add_argument("--curiosity", action="store_true",
                   help="prefer parents from niches that are still improving (old --map_opt)")
    r.add_argument("--size", type=int, default=None,
                   help="render size (default: CLIP input size for 2d, 128 for 3d)")
    r.add_argument("--view-agg", choices=clip_eval.VIEW_AGGREGATIONS, default=None,
                   help="combine a 3d object's per-view scores (default: geomean for "
                        "softmax, which ranks like the old product over views; mean for cosine)")
    g3 = r.add_argument_group("3d options")
    g3.add_argument("--voxels", type=int, default=32, help="voxel grid resolution per axis")
    g3.add_argument("--views", type=int, default=6, help="views per object, 45 degrees apart")
    g3.add_argument("--fixed-bg", action="store_true", help="grey background instead of evolved")
    g3.add_argument("--no-lighting", action="store_true", help="flat colours, no shading")
    g3.add_argument("--march-step", type=float, default=1.0, help="ray-march step in voxels")
    r.add_argument("--device", default="auto",
                   help="auto (cuda > mps > cpu), cuda, cuda:N, mps (Apple silicon) or cpu")
    r.add_argument("--precision", choices=sorted(PRECISIONS), default="fp32",
                   help="CLIP/CoCa weights: fp16 or bf16 are usually much faster on cuda/mps")
    r.add_argument("--log-every", type=int, default=1000)
    r.add_argument("--save-every", type=int, default=10_000)
    r.add_argument("--resume", default=None,
                   help="checkpoint.pkl to continue from (keeps its domain, scoring and "
                        "open-ended settings)")
    go = r.add_argument_group("open-ended niches (2d only)")
    go.add_argument("--open-ended", action="store_true",
                    help="grow the niche set: images far from every niche found new niches "
                         "named by CoCa captions (use --niches none to start from nothing)")
    go.add_argument("--novelty-mode", choices=["text", "image", "both"], default="text",
                    help="novel = far from every niche's text (text), from every archived "
                         "elite's image (image), or both")
    go.add_argument("--novelty-threshold", type=float, default=0.26,
                    help="text novelty: max cosine to all niche texts must be below this")
    go.add_argument("--image-novelty-threshold", type=float, default=0.89,
                    help="image novelty: max cosine to all archived elite images must be below this")
    go.add_argument("--duplicate-threshold", type=float, default=0.95,
                    help="reject a caption whose CLIP text embedding is this close to a niche's")
    go.add_argument("--caption-threshold", type=float, default=0.28,
                    help="minimum CoCa image/caption cosine for a caption to name a niche")
    go.add_argument("--max-new-niches-per-batch", type=int, default=2)
    go.add_argument("--max-niches", type=int, default=None, help="stop adding niches at this many")
    go.add_argument("--caption-model", default=None, help="default: coca_ViT-B-32")
    go.add_argument("--caption-pretrained", default=None, help="default: laion2b_s13b_b90k")
    go.add_argument("--caption-prefix", default="",
                    help='text every caption starts from, e.g. "a photo of a" to steer CoCa '
                         "towards naming objects (stripped from niche names)")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("export", help="render the elites in a checkpoint to PNGs (and meshes)")
    e.add_argument("checkpoint")
    e.add_argument("--out", default=None, help="default: <checkpoint dir>/elites")
    e.add_argument("--size", type=int, default=512)
    e.add_argument("--top", type=int, default=None)
    e.add_argument("--min-score", type=float, default=None)
    e.add_argument("--discovered-only", action="store_true",
                   help="open-ended runs: only export niches found during the run")
    e.add_argument("--mesh", action="store_true",
                   help="3d: also write a coloured .ply mesh per elite (needs scikit-image)")
    e.add_argument("--device", default="auto")
    e.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
