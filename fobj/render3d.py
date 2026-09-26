"""Render 3D CPPN genomes: voxel density + colour, ray-marched from several views.

A headless port of ``legacy/render_vox_fast.py``. The old code extracted the
density = 0.5 isosurface with marching cubes and drew it with fixed-function
OpenGL through a pygame window. Here the same isosurface is found by marching
rays through the trilinearly interpolated voxel grid in torch, so it runs on
CPU or GPU without a display and renders every view of a genome in one batch.
The scene matches the old one: object in [-1, 1]^3, camera at distance 2 with
a 90 degree field of view, yaw steps of 45 degrees with a 5 degree tilt on
every other view, and the same three camera-fixed directional lights (only
the first is specular, as with OpenGL's defaults).

CPPN inputs are ``x, y, z, d, dxz``: coordinates in [-1, 1], distance from the
centre and distance from the vertical (y) axis. Outputs, all squashed to
[0, 1] by a sigmoid:

* 0: density; the surface is where it crosses 0.5 (the grid border is forced
  empty, as in the old code, so every surface is closed);
* 1-3: HSV colour;
* 4-10 (optional): scene parameters, read at the grid centre, the analogue of
  the old hack that evolved them in spare neuron biases/time constants:
  background RGB, shininess, specular, ambient, diffuse.

``marching_cubes_mesh``/``save_ply`` export the isosurface as a coloured mesh
(needs scikit-image).
"""
import math

import torch
import torch.nn.functional as F

from .pytorch_neat import CPPN

LEAF_NAMES = ("x", "y", "z", "d", "dxz")
SURFACE_OUTPUTS = ("density", "hue", "saturation", "value")
SCENE_OUTPUTS = ("bg_r", "bg_g", "bg_b", "shininess", "specular", "ambient", "diffuse")
THRESHOLD = 0.5
DEFAULT_BG = (0.4078, 0.4575, 0.4811)  # the old --fixed_bg colour
DEFAULT_MATERIAL = {"shininess": 0.5, "specular": 0.5, "ambient": 0.3, "diffuse": 0.6}

# Directions towards the lights, in camera space (legacy GL_POSITIONs).
LIGHT_DIRS = ((0.0, 2.0, -1.0), (10.0, -5.0, 20.0), (-10.0, 0.0, 10.0))


def hsv_to_rgb(hsv):
    """hsv: (3, ...) in [0, 1] -> rgb (3, ...)."""
    h, s, v = hsv[0], hsv[1], hsv[2]
    k = torch.stack([(5 + h * 6) % 6, (3 + h * 6) % 6, (1 + h * 6) % 6])
    return v - v * s * torch.clamp(torch.minimum(k, 4 - k), 0, 1)


def _rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])


def default_views(n=6, step=45.0, jitter=5.0):
    """(yaw, tilt) pairs of the old renderer: 0, 45, 90, ... alternating tilt."""
    return [(i * step, jitter if i % 2 else 0.0) for i in range(n)]


class Renderer3D:
    def __init__(self, config, size=224, voxels=32, views=None, evolve_scene=None,
                 fixed_bg=False, lighting=True, device="cpu", camera_distance=2.0, fov=90.0,
                 step=1.0):
        """``step`` is the ray-marching step in voxels (the surface is then refined
        by linear interpolation between the last two samples)."""
        gc = config.genome_config
        if gc.num_inputs != len(LEAF_NAMES):
            raise ValueError(f"3D renderer needs {len(LEAF_NAMES)} inputs, config has {gc.num_inputs}")
        n_out = gc.num_outputs
        if n_out not in (len(SURFACE_OUTPUTS), len(SURFACE_OUTPUTS) + len(SCENE_OUTPUTS)):
            raise ValueError(f"3D renderer needs {len(SURFACE_OUTPUTS)} or "
                             f"{len(SURFACE_OUTPUTS) + len(SCENE_OUTPUTS)} outputs, config has {n_out}")
        has_scene = n_out > len(SURFACE_OUTPUTS)
        self.evolve_scene = has_scene if evolve_scene is None else (evolve_scene and has_scene)
        self.output_names = SURFACE_OUTPUTS + (SCENE_OUTPUTS if has_scene else ())
        self.config = config
        self.size = size
        self.n = voxels
        self.fixed_bg = fixed_bg
        self.lighting = lighting
        self.device = torch.device(device)
        self.views = list(views) if views is not None else default_views()
        self.num_views = len(self.views)

        lin = torch.linspace(-1.0, 1.0, voxels, device=self.device)
        # Volume layout is (z, y, x) so grid_sample's (x, y, z) coordinates line up.
        z, y, x = torch.meshgrid(lin, lin, lin, indexing="ij")
        self.inputs = {"x": x, "y": y, "z": z, "d": torch.sqrt(x**2 + y**2 + z**2),
                       "dxz": torch.sqrt(x**2 + z**2)}
        zero = torch.zeros(1, device=self.device)
        self.centre = {k: zero for k in LEAF_NAMES}
        self.voxel = 2.0 / (voxels - 1)

        # Per-view rays in object space.
        half = math.tan(math.radians(fov) / 2)
        u = torch.linspace(-half, half, size)
        v, u = torch.meshgrid(-u, u, indexing="ij")  # image row 0 is the top (+y)
        d_cam = F.normalize(torch.stack([u, v, -torch.ones_like(u)], -1), dim=-1)  # (H, W, 3)
        o_cam = torch.tensor([0.0, 0.0, camera_distance])
        rots, origins, dirs = [], [], []
        for yaw, tilt in self.views:
            r = _rot_y(yaw) @ _rot_x(tilt)  # object -> camera, as glRotatef(yaw); glRotatef(tilt)
            rots.append(r)
            origins.append(r.T @ o_cam)
            dirs.append(d_cam @ r)  # r.T @ d for each pixel
        self.rots = torch.stack(rots).to(self.device)            # (V, 3, 3)
        self.ray_o = torch.stack(origins).to(self.device)        # (V, 3)
        self.ray_d = torch.stack(dirs).to(self.device)           # (V, H, W, 3)
        self.lights = F.normalize(torch.tensor(LIGHT_DIRS), dim=-1).to(self.device)

        self.step = step * self.voxel

    # -- CPPN -> fields ------------------------------------------------------

    @torch.no_grad()
    def fields(self, genome):
        """Returns (density (N,N,N), rgb (3,N,N,N), scene dict)."""
        cppn = CPPN(genome, self.config, LEAF_NAMES, self.output_names,
                    output_activation=torch.sigmoid)
        out = torch.nan_to_num(cppn(**self.inputs), nan=0.0)
        density = out[0].clone()
        density[[0, -1], :, :] = 0.0
        density[:, [0, -1], :] = 0.0
        density[:, :, [0, -1]] = 0.0
        rgb = hsv_to_rgb(out[1:4].clamp(0, 1))

        scene = {"bg": torch.tensor(DEFAULT_BG, device=self.device), **{
            k: torch.tensor(v, device=self.device) for k, v in DEFAULT_MATERIAL.items()}}
        if self.evolve_scene:
            s = torch.nan_to_num(cppn(**self.centre)[4:, 0], nan=0.5)
            scene.update(shininess=s[3], specular=s[4], ambient=s[5], diffuse=s[6])
            if not self.fixed_bg:
                scene["bg"] = s[:3]
        return density, rgb, scene

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def _sample(volume, pts):
        """Trilinear lookup. volume (C, D, H, W), pts (..., 3) as (x, y, z) in [-1, 1]
        -> (C, ...). Uses ``grid_sample`` where it is native (CPU, CUDA) and a
        plain-indexing version on MPS, where 5-D ``grid_sample`` is missing from
        some PyTorch builds."""
        if volume.device.type == "mps":
            return Renderer3D._trilinear(volume, pts)
        shape = pts.shape[:-1]
        out = F.grid_sample(volume[None], pts.reshape(1, -1, 1, 1, 3), mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        return out.reshape(volume.shape[0], *shape)

    @staticmethod
    def _trilinear(volume, pts):
        """``_sample`` from plain indexing: same result as ``grid_sample`` with
        ``align_corners=True, padding_mode="zeros"``, on any backend."""
        C, D, H, W = volume.shape
        shape = pts.shape[:-1]
        p = pts.reshape(-1, 3)
        size = torch.tensor([W, H, D], device=p.device, dtype=p.dtype)
        u = (p + 1) * 0.5 * (size - 1)            # continuous voxel coords (x, y, z)
        i0 = torch.floor(u)
        f = u - i0
        i0 = i0.long()
        flat = volume.reshape(C, -1)
        out = torch.zeros(C, len(p), device=volume.device, dtype=volume.dtype)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                         * (f[:, 2] if dz else 1 - f[:, 2]))
                    inside = ((ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                              & (iz >= 0) & (iz < D))
                    idx = (iz.clamp(0, D - 1) * H + iy.clamp(0, H - 1)) * W + ix.clamp(0, W - 1)
                    out += flat[:, idx] * (w * inside)
        return out.reshape(C, *shape)

    @torch.no_grad()
    def render_views(self, genome):
        """Returns (num_views, 3, size, size) images in [0, 1]."""
        density, rgb, scene = self.fields(genome)
        return self.shade(density, rgb, scene)

    def render(self, genome):
        """First view only, (3, size, size); for contact sheets."""
        return self.render_views(genome)[0]

    @torch.no_grad()
    def shade(self, density, rgb, scene):
        """Ray-march every view (one at a time to bound memory).

        Only rays through the bounding box of the occupied voxels are marched,
        which makes small objects much cheaper than the full grid.
        """
        V, size = self.num_views, self.size
        bg = scene["bg"].view(3, 1, 1).expand(3, size, size)
        occ = torch.nonzero(density > THRESHOLD)
        if len(occ) == 0:
            return bg[None].expand(V, 3, size, size).clone()
        # (z, y, x) voxel indices -> (x, y, z) box corners, one voxel of margin.
        lo = (occ.amin(0).flip(0) - 1).float() * self.voxel - 1.0
        hi = (occ.amax(0).flip(0) + 1).float() * self.voxel - 1.0
        lo, hi = lo.clamp(-1, 1), hi.clamp(-1, 1)
        return torch.stack([self._shade_view(v, density, rgb, scene, lo, hi, bg)
                            for v in range(V)])

    def _shade_view(self, vi, density, rgb, scene, lo, hi, bg):
        dens = density[None]
        o, d = self.ray_o[vi], self.ray_d[vi].reshape(-1, 3)             # (3,), (P, 3)

        # Ray / bounding-box intersection; march only the rays that hit it.
        inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t0, t1 = (lo - o) * inv, (hi - o) * inv
        t_near = torch.minimum(t0, t1).amax(-1).clamp(min=0.0)
        t_far = torch.maximum(t0, t1).amin(-1)
        rays = torch.nonzero(t_far > t_near)[:, 0]
        img = bg.permute(1, 2, 0).reshape(-1, 3).clone()                  # (P, 3)
        if len(rays) == 0:
            return img.reshape(self.size, self.size, 3).permute(2, 0, 1)
        d, t_near, t_far = d[rays], t_near[rays], t_far[rays]
        n_steps = int(math.ceil((t_far - t_near).max().item() / self.step)) + 1

        # March: find the first sample above the threshold along each ray.
        ks = torch.arange(n_steps, device=self.device, dtype=torch.float32)
        t = t_near[:, None] + ks * self.step                              # (R, S)
        pts = o + t[..., None] * d[:, None, :]                            # (R, S, 3)
        vals = self._sample(dens, pts)[0].masked_fill(t > t_far[:, None], 0.0)
        inside = vals > THRESHOLD
        hit = inside.any(-1)
        k = inside.to(torch.uint8).argmax(-1)                             # first crossing
        k_prev = (k - 1).clamp(min=0)
        v1 = vals.gather(-1, k[:, None])[:, 0]
        v0 = vals.gather(-1, k_prev[:, None])[:, 0]
        frac = ((THRESHOLD - v0) / (v1 - v0).clamp(min=1e-6)).clamp(0, 1)
        frac = torch.where(k > 0, frac, torch.zeros_like(frac))
        t_hit = t_near + (k_prev.float() + frac) * self.step
        rays, d, t_hit = rays[hit], d[hit], t_hit[hit]
        p = o + t_hit[:, None] * d                                        # (R', 3)

        colour = self._sample(rgb, p).T                                   # (R', 3)
        if self.lighting:
            offs = torch.eye(3, device=self.device) * self.voxel
            grad = torch.stack([self._sample(dens, p + offs[i])[0] - self._sample(dens, p - offs[i])[0]
                                for i in range(3)], -1)
            n = F.normalize(-grad, dim=-1) @ self.rots[vi].T              # outward normal, camera space
            ndotl = (n[:, None, :] * self.lights).sum(-1).clamp(min=0)    # (R', L)
            diffuse = scene["diffuse"] * ndotl.sum(-1, keepdim=True)
            half = F.normalize(self.lights[0] + torch.tensor([0.0, 0.0, 1.0], device=self.device), dim=-1)
            spec_angle = (n * half).sum(-1, keepdim=True).clamp(min=0)
            shininess = 1.0 + scene["shininess"] * 127.0
            specular = scene["specular"] * spec_angle ** shininess * (ndotl[:, :1] > 0)
            colour = colour * (scene["ambient"] + diffuse) + specular
        img[rays] = colour.clamp(0, 1)
        return img.reshape(self.size, self.size, 3).permute(2, 0, 1)


def marching_cubes_mesh(renderer, genome):
    """Isosurface as (verts (M, 3) in [-1, 1], faces (F, 3), colours (M, 3) in [0, 1])."""
    from skimage import measure

    density, rgb, _ = renderer.fields(genome)
    vol = density.cpu().numpy()
    if vol.max() <= THRESHOLD:
        return None
    verts, faces, _, _ = measure.marching_cubes(vol, THRESHOLD)
    # skimage returns (z, y, x) index coordinates.
    xyz = torch.from_numpy(verts[:, ::-1].copy()).float() * renderer.voxel - 1.0
    cols = Renderer3D._sample(rgb.cpu(), xyz).T
    # Flip winding so faces point outwards in (x, y, z) order.
    return xyz.numpy(), faces[:, ::-1].copy(), cols.clamp(0, 1).numpy()


def save_ply(path, verts, faces, colours):
    """Binary little-endian PLY with per-vertex colour."""
    import numpy as np

    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v = np.empty(len(verts), dtype=vdt)
    v["x"], v["y"], v["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    c = (np.asarray(colours) * 255).round().astype(np.uint8)
    v["red"], v["green"], v["blue"] = c[:, 0], c[:, 1], c[:, 2]
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    f = np.empty(len(faces), dtype=fdt)
    f["n"], f["i"] = 3, faces
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(v)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(v.tobytes())
        fh.write(f.tobytes())
