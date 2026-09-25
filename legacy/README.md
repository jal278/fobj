# Legacy code (Python 2, MultiNEAT, Caffe)

This is the original fooling-objects code, kept for reference. It does not run
on a modern stack. The `fobj` package replaces both paths:

* the 2D image path (`fool.py --2d`, `melites.py`, `image_rec.py`) is
  `fobj run --domain 2d`;
* the 3D voxel path (`fool.py`, `render_vox*.py`, `fool_eval.pyx`,
  `saveply.py`) is `fobj run --domain 3d` and `fobj export --mesh`.

`nodecalc/` holds the ImageNet synset ids and the WordNet-hierarchy niche
matrix used by the old `--wordnet` option.
