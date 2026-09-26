"""Device and precision selection for the image/text models.

``auto`` picks CUDA, then Apple-silicon MPS, then CPU. On macOS, ``fobj``
also turns on PyTorch's MPS CPU fallback (see ``fobj/__init__.py``), so an
operator MPS lacks in a given PyTorch version runs on the CPU instead of
crashing the run.
"""
import torch

PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def mps_available():
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def pick_device(name="auto"):
    """``auto`` | ``cuda`` | ``cuda:N`` | ``mps`` | ``cpu`` -> a device string."""
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {name}: CUDA is not available in this PyTorch build")
    if name == "mps" and not mps_available():
        built = getattr(torch.backends.mps, "is_built", lambda: False)()
        why = ("this macOS/hardware does not support it (needs Apple silicon or an AMD GPU "
               "and macOS 12.3+)" if built else "this PyTorch build has no MPS support")
        raise SystemExit(f"--device mps: MPS is not available: {why}")
    return name


def model_dtype(precision, device):
    """Weights/activations dtype for the CLIP and CoCa models.

    Half precision roughly halves memory and is usually much faster on GPUs,
    including Apple silicon; on CPU it is slow or unsupported, so it is
    refused there. Scores are always returned as float32/float64.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; use one of {sorted(PRECISIONS)}")
    dtype = PRECISIONS[precision]
    if dtype != torch.float32 and torch.device(device).type == "cpu":
        raise SystemExit(f"--precision {precision} is for GPUs (cuda/mps); use fp32 on CPU")
    return dtype
