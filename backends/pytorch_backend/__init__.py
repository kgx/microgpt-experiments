"""
PyTorch training and inference backend. Same math as Python/NumPy backends.
Requires torch. Install with: pip install microgpt[pytorch]
"""

from .pytorch_backend import PyTorchBackend
from .inference import load_model_pytorch, gpt_step_pytorch
from .forward import forward_batched, forward_step

__all__ = [
    "PyTorchBackend",
    "load_model_pytorch",
    "gpt_step_pytorch",
    "forward_batched",
    "forward_step",
]
