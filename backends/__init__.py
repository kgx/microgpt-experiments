"""
Pluggable training backends. Each backend implements the same contract so
train.py can run with "python" (default), or "numpy" when numpy is installed.
"""

from backends.python_backend import PythonBackend

_REGISTRY: dict[str, type] = {
    "python": PythonBackend,
}

try:
    from backends.numpy_backend import NumPyBackend
    _REGISTRY["numpy"] = NumPyBackend
except ImportError:
    pass


def register_backend(name: str, backend_class: type) -> None:
    """Register a backend class under the given name."""
    _REGISTRY[name] = backend_class


def get_backend(name: str):
    """Return an instance of the backend with the given name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()
