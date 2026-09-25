# Legacy code (Python 2, MultiNEAT, Caffe)

This is the original fooling-objects code, kept for reference. It does not run
on a modern stack. The 2D image path (`fool.py --2d`, `melites.py`,
`image_rec.py`) has been replaced by the `fobj` package. The 3D voxel path
(`render_vox*.py`, `fool_eval.pyx`, `saveply.py`) has not been ported yet.
`nodecalc/` holds the ImageNet synset ids and the WordNet-hierarchy niche
matrix used by the old `--wordnet` option.
