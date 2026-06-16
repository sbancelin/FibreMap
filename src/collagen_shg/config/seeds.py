"""Reproducible randomness — master seed → independent named child generators (phase0 §8).

One master seed per run; independent child streams are derived per component (``structure``,
``noise``, ...) via ``numpy.random.SeedSequence`` / ``PCG64`` so the components are both
independent and reproducible. Children are keyed by **name** (not draw order), so adding a new
component never perturbs existing streams. All effective seeds are recorded for provenance.
"""

from __future__ import annotations

import hashlib

from numpy.random import PCG64, Generator, SeedSequence

__all__ = ["SeedManager", "derive_generator", "name_key"]


def name_key(name: str) -> int:
    """Stable 64-bit integer key for a child name (independent of Python's hash randomization)."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def derive_generator(master_seed: int, name: str) -> Generator:
    """A reproducible, name-independent ``Generator`` for ``(master_seed, name)``."""
    ss = SeedSequence(entropy=int(master_seed), spawn_key=(name_key(name),))
    return Generator(PCG64(ss))


class SeedManager:
    """Derives named child generators from a master seed and logs them for provenance."""

    def __init__(self, master_seed: int) -> None:
        self.master_seed = int(master_seed)
        self._log: dict[str, int] = {}

    def seed_sequence(self, name: str) -> SeedSequence:
        key = name_key(name)
        self._log[name] = key
        return SeedSequence(entropy=self.master_seed, spawn_key=(key,))

    def generator(self, name: str) -> Generator:
        """Return an independent ``PCG64`` generator for the named component."""
        return Generator(PCG64(self.seed_sequence(name)))

    def provenance(self) -> dict[str, object]:
        """Record of the RNG, master seed, and every child key handed out so far."""
        return {
            "rng": "PCG64",
            "master_seed": self.master_seed,
            "children": dict(self._log),
        }