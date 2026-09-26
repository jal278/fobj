import pytest
import torch

from fobj.device import PRECISIONS, model_dtype, pick_device


def test_pick_device():
    assert pick_device("cpu") == "cpu"
    assert pick_device("auto") in ("cuda", "mps", "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="CUDA"):
            pick_device("cuda")
    if not (torch.backends.mps.is_available()):
        with pytest.raises(SystemExit, match="MPS"):
            pick_device("mps")


def test_model_dtype():
    assert model_dtype("fp32", "cpu") == torch.float32
    with pytest.raises(SystemExit, match="GPU"):
        model_dtype("fp16", "cpu")
    assert model_dtype("fp16", "mps") == torch.float16
    assert model_dtype("bf16", "cuda") == torch.bfloat16
    with pytest.raises(ValueError):
        model_dtype("fp8", "cuda")
    assert set(PRECISIONS) == {"fp32", "fp16", "bf16"}
