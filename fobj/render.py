"""Render 2D CPPN genomes to RGB images with the vendored PyTorch-NEAT CPPN."""
import torch

from .pytorch_neat import CPPN

LEAF_NAMES = ("x", "y", "d")
OUTPUT_NAMES = ("r", "g", "b")


def _output_activation(x):
    # Map an unbounded output to a colour channel in [0, 1].
    return torch.sigmoid(x)


class Renderer2D:
    """Evaluates a genome at every pixel of a ``size`` x ``size`` grid."""

    def __init__(self, config, size=224, device="cpu"):
        gc = config.genome_config
        if gc.num_inputs != len(LEAF_NAMES) or gc.num_outputs != len(OUTPUT_NAMES):
            raise ValueError(
                f"2D renderer needs {len(LEAF_NAMES)} inputs and {len(OUTPUT_NAMES)} "
                f"outputs, config has {gc.num_inputs} and {gc.num_outputs}")
        self.config = config
        self.size = size
        self.device = torch.device(device)
        lin = torch.linspace(-1.0, 1.0, size, device=self.device)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        d = torch.sqrt(x**2 + y**2)
        self.inputs = {"x": x, "y": y, "d": d}

    @torch.no_grad()
    def render(self, genome):
        """Returns a float tensor of shape (3, size, size) in [0, 1]."""
        cppn = CPPN(genome, self.config, LEAF_NAMES, OUTPUT_NAMES,
                    output_activation=_output_activation)
        img = cppn(**self.inputs)
        return torch.nan_to_num(img, nan=0.5).clamp_(0.0, 1.0)

    def render_batch(self, genomes):
        return torch.stack([self.render(g) for g in genomes])


def to_pil(img):
    """(3, H, W) float tensor in [0, 1] -> PIL image."""
    from PIL import Image

    arr = (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)
