"""Niche definitions: one text label per MAP-Elites niche."""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
IMAGENET_TSV = DATA / "imagenet_wordnet.tsv"


def imagenet_classes():
    """The 1000 ILSVRC-2012 classes as rows of wnid/synset/name/lemmas."""
    with open(IMAGENET_TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_niches(spec="imagenet"):
    """Returns a list of niche names.

    ``spec`` is ``"imagenet"`` (the 1000 ImageNet WordNet classes, the niches
    of the original fooling-objects experiments) or a path to a text file
    with one name per line (blank lines and ``#`` comments ignored).
    """
    if spec == "imagenet":
        return [row["name"] for row in imagenet_classes()]
    names = []
    for line in Path(spec).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    if not names:
        raise ValueError(f"no niche names found in {spec}")
    return names
