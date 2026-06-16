"""Metrics quickstart (Livrable 1): run a few metrics on synthetic patterns.

    python examples/metrics_demo.py
"""

from __future__ import annotations

import numpy as np

from collagen_shg import metrics
from collagen_shg.representations import conventions as cv


def main() -> int:
    n, lam = 256, 8.0
    yy, xx = np.mgrid[0:n, 0:n]
    # stripes varying along x -> fibres run along y (phi = pi/2); spacing = lam
    image = np.sin(2 * np.pi * xx / lam)

    st = metrics.structure_tensor_2d(image, sigma=1.0, rho=4.0)
    interior = (slice(32, -32), slice(32, -32))
    op = metrics.order_parameter_2d(st.orientation[interior], st.coherence[interior])
    ps = metrics.power_spectrum_orientation(image)

    print("# structure tensor + order parameter on x-stripes (expect phi=90 deg, S2~1)")
    print(f"    S2={op.S2:.3f}  theta_bar={np.rad2deg(op.theta_bar):.1f} deg  kappa={op.kappa:.1f}")
    print("# Fourier power spectrum (expect orientation=90 deg, spacing=8)")
    print(f"    orientation={np.rad2deg(ps.orientation):.1f} deg  spacing={ps.spacing:.2f}")

    # a +1/2 topological disclination
    c = n / 2 - 0.5
    theta = cv.wrap_axial(0.5 * np.arctan2(yy - c, xx - c))
    dd = metrics.defect_density(theta)
    print("# defect density on a +1/2 disclination (expect 1 defect)")
    print(f"    n_defects={dd.n_defects}  total_charge={dd.total_charge:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
