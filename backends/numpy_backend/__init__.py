"""
NumPy training backends. Same math as the Python backend but with vectorized ops
and manual backward. Requires numpy; uses numba for hot loops if available.
"""

from .numpy_sequential import NumPyBackend
from .numpy_batched import NumPyBatchedBackend
from .shared import (
    verify_numba_jit,
    _HAS_NUMBA,
    _attention_scores_numba,
    _attention_out_numba,
)

__all__ = [
    "NumPyBackend",
    "NumPyBatchedBackend",
    "verify_numba_jit",
    "_HAS_NUMBA",
    "_attention_scores_numba",
    "_attention_out_numba",
]
