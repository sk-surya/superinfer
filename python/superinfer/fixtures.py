"""Small model-independent fixtures for reference and property tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorFixture:
    """Immutable flat tensor fixture with explicit shape and values."""

    shape: tuple[int, ...]
    values: tuple[float, ...]


def make_identity_fixture(size: int) -> TensorFixture:
    """Create a square identity fixture, rejecting non-positive dimensions."""

    if size <= 0:
        raise ValueError("fixture size must be positive")
    values = tuple(float(row == column) for row in range(size) for column in range(size))
    return TensorFixture(shape=(size, size), values=values)

