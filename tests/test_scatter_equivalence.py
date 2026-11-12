"""Is torch's native scatter_reduce a drop-in for the one torch_scatter call?

Upstream `trainers/mv_utils_zs.py` line 129 is the only place `torch_scatter` is
used:

    grid = scatter(_z, coordinates.long(), dim=1, out=grid, reduce="max")

`torch-scatter==2.0.9` is the single dependency that needs compiling, and it is
the likely failure point on Windows with a current PyTorch. If it will not
install, that call is replaced with

    grid.scatter_reduce_(1, index, src, reduce="amax", include_self=True)

These tests are the evidence that the substitution is faithful. They run only
when torch is installed, and the torch_scatter comparison runs only when
torch_scatter is installed too - so this file is informative on a machine where
the original dependency works, and still checks the replacement's semantics on
one where it does not.

If the substitution is used in a reported run, it is a deviation: log it in
REPRO_LOG.md and carry it into the report.
"""

import pytest

torch = pytest.importorskip("torch")


def _case(seed):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(2, 64, generator=g)
    index = torch.randint(0, 8, (2, 64), generator=g)
    out = torch.full((2, 8), -float("inf"))
    return src, index, out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_native_scatter_reduce_matches_a_reference_max(seed):
    """scatter_reduce_ with amax really computes a per-bucket maximum."""
    src, index, out = _case(seed)
    got = out.clone().scatter_reduce_(1, index, src, reduce="amax", include_self=True)

    expected = torch.full_like(out, -float("inf"))
    for row in range(src.shape[0]):
        for i in range(src.shape[1]):
            b = int(index[row, i])
            expected[row, b] = max(float(expected[row, b]), float(src[row, i]))
    assert torch.equal(got, expected)


def test_empty_buckets_keep_the_initial_value():
    """The upstream call passes `out=grid`, so untouched cells must survive.

    This is what `include_self=True` controls, and getting it wrong would change
    the rendered depth maps for sparse regions - silently, and only for sparse
    clouds, which is exactly the robot case.
    """
    src = torch.tensor([[1.0, 2.0]])
    index = torch.tensor([[0, 0]])
    out = torch.tensor([[-5.0, -5.0, -5.0]])
    got = out.clone().scatter_reduce_(1, index, src, reduce="amax", include_self=True)
    assert got.tolist() == [[2.0, -5.0, -5.0]]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_torch_scatter_when_it_is_available(seed):
    """The real comparison, when the original dependency is installed."""
    scatter = pytest.importorskip("torch_scatter").scatter
    src, index, out = _case(seed)
    reference = scatter(src, index, dim=1, out=out.clone(), reduce="max")
    replacement = out.clone().scatter_reduce_(1, index, src, reduce="amax", include_self=True)
    assert torch.allclose(reference, replacement)
