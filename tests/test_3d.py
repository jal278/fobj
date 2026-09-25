import colorsys
import random

import numpy as np
import pytest
import torch

from fobj.clip_eval import aggregate_views
from fobj.genome import GenomeFactory, load_config
from fobj.render import DEFAULT_CONFIGS, make_renderer
from fobj.render3d import Renderer3D, hsv_to_rgb, marching_cubes_mesh, save_ply
from fobj.genome import DEFAULT_CONFIG

CFG3D = DEFAULT_CONFIG.parent / DEFAULT_CONFIGS["3d"]
SCENE = {"bg": torch.tensor([0.1, 0.2, 0.3]), "shininess": torch.tensor(0.5),
         "specular": torch.tensor(0.0), "ambient": torch.tensor(0.3),
         "diffuse": torch.tensor(0.6)}


def sphere(r, radius=0.5):
    dens = torch.sigmoid(20 * (radius - r.inputs["d"]))
    rgb = torch.ones(3, *dens.shape)
    return dens, rgb


def test_hsv_to_rgb_matches_colorsys():
    hsv = torch.rand(3, 50)
    got = hsv_to_rgb(hsv).T.numpy()
    want = np.array([colorsys.hsv_to_rgb(*c) for c in hsv.T.tolist()])
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_sphere_render():
    r = Renderer3D(load_config(CFG3D), size=64, voxels=24, views=[(0, 0), (90, 0), (45, 0)])
    imgs = r.shade(*sphere(r), SCENE)
    assert imgs.shape == (3, 3, 64, 64)
    bg = SCENE["bg"]
    # Corners miss, centre hits and is lit (not background).
    torch.testing.assert_close(imgs[:, :, 0, 0], bg.expand(3, 3))
    assert (imgs[:, :, 32, 32] - bg).abs().sum(-1).min() > 0.1
    # A sphere looks the same from every yaw.
    torch.testing.assert_close(imgs[0], imgs[1], atol=0.03, rtol=0)
    # (the diagonal view sees the voxel grid at 45 degrees: close, not exact)
    assert (imgs[0] - imgs[2]).abs().mean() < 0.005
    # Silhouette radius ~ what the pinhole camera predicts for r=0.5 at distance 2.
    hit = (imgs[0] - bg[:, None, None]).abs().sum(0) > 1e-3
    frac = hit.float().mean().item()
    tan_half = 0.5 / np.sqrt(2**2 - 0.5**2)  # tangent of the angular radius
    assert abs(frac - np.pi * tan_half**2 / 4) < 0.01


def test_empty_volume_is_background():
    r = Renderer3D(load_config(CFG3D), size=16, voxels=8)
    imgs = r.shade(torch.zeros(8, 8, 8), torch.ones(3, 8, 8, 8), SCENE)
    torch.testing.assert_close(imgs, SCENE["bg"].view(1, 3, 1, 1).expand(6, 3, 16, 16))


def test_random_genomes_render_and_mesh(tmp_path):
    random.seed(0)
    config = load_config(CFG3D)
    r = make_renderer({"domain": "3d", "size": 32, "voxels": 16}, config)
    factory = GenomeFactory(config)
    meshes = 0
    for _ in range(10):
        g = factory.new()
        for _ in range(5):
            g = factory.mutate(g)
        imgs = r.render_views(g)
        assert imgs.shape == (6, 3, 32, 32)
        assert torch.isfinite(imgs).all() and imgs.min() >= 0 and imgs.max() <= 1
        pytest.importorskip("skimage")
        m = marching_cubes_mesh(r, g)
        if m is not None:
            verts, faces, cols = m
            assert np.abs(verts).max() <= 1 + 1e-5 and faces.max() < len(verts)
            save_ply(tmp_path / "m.ply", *m)
            assert (tmp_path / "m.ply").read_bytes().startswith(b"ply\n")
            meshes += 1
    assert meshes > 0


def test_aggregate_views():
    s = np.array([[[0.5, 0.1], [0.2, 0.4]]])  # 1 object, 2 views, 2 niches
    np.testing.assert_allclose(aggregate_views(s, "mean"), [[0.35, 0.25]])
    np.testing.assert_allclose(aggregate_views(s, "min"), [[0.2, 0.1]])
    np.testing.assert_allclose(aggregate_views(s, "prod"), [[0.1, 0.04]])
    np.testing.assert_allclose(aggregate_views(s, "geomean"), [[0.1**0.5, 0.2]])
    np.testing.assert_allclose(aggregate_views(s[:, :1], "geomean"), s[:, 0])
    with pytest.raises(ValueError):
        aggregate_views(-s, "geomean")
