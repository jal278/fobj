"""Regenerate fobj/data/imagenet_wordnet.tsv from the 1000 ILSVRC-2012 WordNet ids.

Each row maps an ImageNet class index to its WordNet synset and a short,
human-readable name used to build the CLIP text prompt for that niche.
First lemmas that collide (``crane``, ``maillot``) are disambiguated with
their hypernym, e.g. ``crane (wading bird)``.

Usage:
    NLTK_ALLOW_PROXIED_URLOPEN=1 python scripts/make_imagenet_niches.py
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import nltk

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synsets", default=ROOT / "legacy" / "nodecalc" / "synsets.txt")
    ap.add_argument("--out", default=ROOT / "fobj" / "data" / "imagenet_wordnet.tsv")
    args = ap.parse_args()

    nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet as wn

    wnids = [line.strip() for line in open(args.synsets) if line.strip()]
    synsets = [wn.synset_from_pos_and_offset("n", int(w[1:])) for w in wnids]

    def pretty(s):
        return s.replace("_", " ")

    first = [pretty(s.lemma_names()[0]) for s in synsets]
    counts = Counter(first)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["index", "wnid", "synset", "name", "lemmas"])
        for i, (wnid, s, name) in enumerate(zip(wnids, synsets, first)):
            if counts[name] > 1:
                hyper = s.hypernyms()
                if hyper:
                    name = f"{name} ({pretty(hyper[0].lemma_names()[0])})"
            lemmas = ", ".join(pretty(l) for l in s.lemma_names())
            w.writerow([i, wnid, s.name(), name, lemmas])
    print(f"wrote {len(wnids)} classes to {args.out}")


if __name__ == "__main__":
    main()
