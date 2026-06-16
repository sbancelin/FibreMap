"""Coherent SHG forward/backward demo (Livrable 2, Tier 3).

    python examples/coherent_fb_demo.py

Shows that a uniform-polarity bulk is forward-dominant while a quasi-phase-matched (alternating)
polarity arrangement boosts the backward signal.
"""

from __future__ import annotations

import numpy as np

from collagen_shg.config.models import MicroscopeConfig
from collagen_shg.imaging import CoherentImager
from collagen_shg.representations.phantom import DirectorFields, Phantom, PhantomMeta


def _phantom(polarity, dz):
    shape = polarity.shape
    meta = PhantomMeta(shape_zyx=shape, voxel_size_zyx=(dz, 0.2, 0.2))
    fields = DirectorFields(
        director=np.zeros((3, *shape), dtype=np.float32),
        order_S=np.zeros(shape, dtype=np.float32),
        density=np.ones(shape, dtype=np.float32),
        polarity=polarity.astype(np.float32),
    )
    return Phantom(meta=meta, fields=fields)


def main() -> int:
    n_med, lam = 1.33, 0.9
    dk_b = 8 * np.pi * n_med / lam
    dz = (2 * np.pi / dk_b) / 4  # resolve the QPM period
    nz = 240
    z = np.arange(nz) * dz
    shape = (nz, 2, 2)

    imager = CoherentImager()
    mic = MicroscopeConfig(wavelength_nm=900)

    uniform = imager.fields(_phantom(np.ones(shape), dz), mic)
    qpm_pol = (np.sign(np.sin(dk_b * z))[:, None, None] * np.ones(shape))
    qpm = imager.fields(_phantom(qpm_pol, dz), mic)

    print("# forward/backward ratio (higher = more forward-dominant)")
    print(f"    uniform-polarity bulk : F/B = {uniform.fb_ratio:.3e}")
    print(f"    QPM (alternating)     : F/B = {qpm.fb_ratio:.3e}")
    print(f"    -> QPM enhances backward (lower F/B): {qpm.fb_ratio < uniform.fb_ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
