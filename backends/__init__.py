"""
Pluggable training backends. Each backend implements the same contract so
train.py can run with "python" (default), or "numpy" when numpy is installed.
"""

from backends.python_backend import PythonBackend

_REGISTRY: dict[str, type] = {
    "python": PythonBackend,
}

try:
    from backends.numpy_backend import NumPyBackend, NumPyBatchedBackend
    _REGISTRY["numpy"] = NumPyBackend
    _REGISTRY["numpy_batched"] = NumPyBatchedBackend
except ImportError:
    pass

try:
    from backends.pytorch_backend import PyTorchBackend
    _REGISTRY["pytorch"] = PyTorchBackend
except ImportError:
    pass


def register_backend(name: str, backend_class: type) -> None:
    """Register a backend class under the given name."""
    _REGISTRY[name] = backend_class


def get_backend(name: str):
    """Return an instance of the backend with the given name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        hint = ""
        if name in ("numpy", "numpy_batched"):
            hint = " Install with: pip install microgpt[numpy] or uv sync --extra numpy"
        elif name == "pytorch":
            hint = " Install with: pip install microgpt[pytorch] or uv sync --extra pytorch"
        raise KeyError(f"Unknown backend: {name}. Available: {list(_REGISTRY.keys())}.{hint}")
    return _REGISTRY[name]()
