"""
Verify that Numba JIT is active for the numpy backend and optionally run a tiny benchmark.
Run: python scripts/verify_numba_jit.py
     MICROGPT_VERIFY_JIT=1 python scripts/benchmark_train.py --scenario alice --steps 1 --backend numpy
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

def main():
    try:
        from backends.numpy_backend import verify_numba_jit, _HAS_NUMBA
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from project root and ensure numpy backend is available (uv sync --extra numpy).")
        return 1

    print("Numba available:", _HAS_NUMBA)
    ok, msg = verify_numba_jit()
    print("JIT verification:", msg)
    if ok:
        # After verify, signatures should be non-empty
        from backends.numpy_backend import _attention_scores_numba, _attention_out_numba
        sig_scores = getattr(_attention_scores_numba, "signatures", [])
        sig_out = getattr(_attention_out_numba, "signatures", [])
        print("  _attention_scores_numba signatures:", len(sig_scores))
        print("  _attention_out_numba signatures:", len(sig_out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
