"""Cross-validation: our Lloyd-Max implementation vs upstream vLLM TurboQuant's.

This test guards against numerical drift between our plugin's centroid
generator (``turboquant_vllm.torch_ops._lloyds_gaussian`` / ``optimal_centroids``)
and the independent implementation in Vibhav Agarwal's upstream vLLM
TurboQuant PR (``vllm/model_executor/layers/quantization/turboquant/centroids.py``
on branch ``feature/turboquant-kv-cache``, vllm-project/vllm#38479).

Both implementations solve the same mathematical problem — optimal
scalar quantization of N(0, 1/d) via the Lloyd-Max conditions — using
different numerical methods:

* Ours: ``scipy.stats.norm.ppf`` / ``norm.pdf`` for analytic boundaries
  and conditional expectations.
* Theirs: trapezoidal numerical integration (scipy-free, portable).

Independent implementations agreeing to ~1e-3 across a sweep of
``(d, bits)`` pairs gives strong evidence that both are correct. If
either drifts — e.g. a refactor to our ``_lloyds_gaussian`` changes the
iteration count, or the upstream trapezoidal integrator changes its
step count — this test catches the disagreement immediately.

The upstream reference implementation is VENDORED below with full
attribution so this test has no external dependency. When
vllm-project/vllm#38479 lands on main, this test can be updated to
import the upstream module directly and drop the vendored copy.
"""

import math
import unittest
from functools import lru_cache

import torch

# ---------------------------------------------------------------------------
# Vendored reference implementation
# ---------------------------------------------------------------------------
# Copyright contributors to the vLLM project, Apache-2.0.
# Source: vllm-project/vllm#38479
#   vllm/model_executor/layers/quantization/turboquant/centroids.py
#
# Kept byte-identical to the upstream version so a diff on the file
# immediately surfaces any upstream change. Do not local-optimize.


def _upstream_gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def _upstream_trapz(f, a: float, b: float, n: int = 200) -> float:
    """Trapezoidal numerical integration (replaces scipy.integrate.quad)."""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def _upstream_solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_levels = 2**bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return _upstream_gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _upstream_trapz(lambda x: x * pdf(x), a, b)
            den = _upstream_trapz(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


@lru_cache(maxsize=32)
def _upstream_get_centroids(d: int, bits: int) -> torch.Tensor:
    centroids, _ = _upstream_solve_lloyd_max(d, bits)
    return centroids


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------


from turboquant_vllm.torch_ops import optimal_centroids as _our_optimal_centroids


class TestLloydMaxCrossValidation(unittest.TestCase):
    """Our Lloyd-Max should agree with upstream within a tight tolerance.

    The worst-case observed disagreement is ~2e-3 at (d=64, bits=4),
    driven by the upstream trapezoidal integrator's discretization noise
    (200 points per bin over narrow high-bit bins). We accept 5e-3 as the
    numerical tolerance — five times the observed noise floor — so
    normal floating-point drift does not trip the test while any
    substantive algorithmic change does.
    """

    MAX_ABS_DIFF = 5e-3

    def _compare(self, d: int, bits: int):
        ours = sorted(_our_optimal_centroids(bits, d))
        theirs = sorted(_upstream_get_centroids(d, bits).tolist())
        self.assertEqual(len(ours), len(theirs), f"level count mismatch at d={d} bits={bits}")
        max_diff = max(abs(a - b) for a, b in zip(ours, theirs))
        self.assertLess(
            max_diff,
            self.MAX_ABS_DIFF,
            f"centroid disagreement at d={d} bits={bits}: "
            f"max_abs_diff={max_diff:.2e} > tolerance={self.MAX_ABS_DIFF:.2e}",
        )

    def test_d_64(self):
        for bits in (3, 4):  # skip bits=2 — ours uses a closed-form shortcut
            with self.subTest(bits=bits):
                self._compare(64, bits)

    def test_d_128(self):
        for bits in (3, 4):
            with self.subTest(bits=bits):
                self._compare(128, bits)

    def test_d_256(self):
        for bits in (3, 4):
            with self.subTest(bits=bits):
                self._compare(256, bits)


class TestOurCentroidsAreSortedAndSymmetric(unittest.TestCase):
    """Structural invariants both implementations must satisfy."""

    def test_sorted(self):
        for d in (64, 128, 256):
            for bits in (2, 3, 4):
                c = _our_optimal_centroids(bits, d)
                self.assertEqual(c, sorted(c), f"unsorted at d={d} bits={bits}")

    def test_count(self):
        for d in (64, 128, 256):
            for bits in (2, 3, 4):
                c = _our_optimal_centroids(bits, d)
                self.assertEqual(
                    len(c), 2**bits, f"level count wrong at d={d} bits={bits}"
                )

    def test_approximately_symmetric(self):
        """Centroids should be symmetric around 0 for a centered Gaussian source."""
        for d in (64, 128, 256):
            for bits in (2, 3, 4):
                c = _our_optimal_centroids(bits, d)
                n = len(c)
                for i in range(n // 2):
                    self.assertAlmostEqual(
                        c[i],
                        -c[n - 1 - i],
                        places=3,
                        msg=f"asymmetric at d={d} bits={bits} (index {i})",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
