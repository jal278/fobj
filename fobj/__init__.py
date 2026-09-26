"""fobj: evolving images with CPPN-NEAT + MAP-Elites, with niches defined by CLIP."""
import os
import sys

if sys.platform == "darwin":
    # Run any operator that Apple's MPS backend lacks (in a given PyTorch
    # version) on the CPU instead of raising. Must be set before torch loads.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
