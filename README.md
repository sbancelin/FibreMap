# collagen-shg (FibreMap)

Quantification and comparison of **collagen organization** in connective tissues from
**SHG microscopy** images, built around a *closed validation loop*: a synthetic-image
generator with **known ground truth** is co-designed with the organization **metrics**, so
that analyzers can be measured for bias/variance against the truth, and synthetic images
provide free annotations to train extractors.

This repository implements **Phase 0** (shared infrastructure and representations), **Livrable 1**
(the organization `metrics`, families A–G), and **Livrable 2** (the structure/image generators
and the closed validation loop). The remaining computational modules (resolved analysis, P-SHG)
are present as **stubs with stable interfaces**; they land in Livrables 3–4.

## Livrable 2 — generators + closed loop

- `structure_generator` — `ProceduralStructureGenerator` (Tier 0): places fibrils (von Mises
  axial orientation, crimp, diameter distribution), rasterizes a density + director volume, and
  records the **known** organization ground truth (S₂/S₃/orientation, measured with the
  Livrable 1 metrics). Deterministic for `{config, seed}`.
- `imaging` — `IncoherentImager` (Tier 1): density ⊛ Gaussian PSF (NA/λ) + Beer–Lambert depth
  attenuation + Poisson/read noise. `CoherentImager` (Tier 3, first-order): forward/backward
  SHG via axial phase summation (QPM picture).
- `validation` — `run_closed_loop(config)`: generate → image → analyze → `compare` to the ground
  truth (bias). Aligned phantoms recover their mean orientation; isotropic phantoms measure low
  order.
- GUI — `generate_bundle(config)` and an orientation-RGB layer; `collagen-shg-gui --generate
  config.yaml`.

## Livrable 1 — organization metrics (`collagen_shg.metrics`)

Families A–G, each with analytical tests (uniform → S = 1, isotropic → S = 0, sinusoid →
known orientation/spacing):

- **A** structure tensor — `structure_tensor_2d/3d` (orientation + coherence/FA; fibre axis =
  minor eigenvector)
- **B** order parameters — `order_parameter_2d` (S₂, κ via doubled angle), `order_tensor_3d`
  (S₃, Saupe Q)
- **C** orientation correlation — `orientation_correlation` (ξ via FFT/Wiener–Khinchin)
- **D** Fourier — `power_spectrum_orientation` (orientation + spacing Λ*)
- **E** texture — `glcm_features`, `lbp_histogram`, `gabor_energy`
- **F** per-fibre — `fiber_metrics`, `persistence_length` (crimp)
- **G** topological defects — `defect_density` (winding number)

The closed-loop scoring (generator ↔ metrics) is wired in Livrable 2 via
`collagen_shg.validation.run_closed_loop`; the full SNR/depth/dispersion sweeps build on it.

## Phase 0 scope (this milestone)

- `representations` — `Phantom` (ground truth) and `ImageBundle` data models (Pydantic v2),
  `conventions` (axes/angles/units, fully tested), and `io` (reproducible *bundle* on disk:
  OME-Zarr / OME-TIFF / Parquet / JSON / YAML) with a bit-exact round-trip test.
- `config` — typed Pydantic configuration, YAML loading, tissue/microscope presets.
- seed management — master seed → independent named child seeds (`numpy.random.SeedSequence` /
  `PCG64`), logged in provenance.
- a `validation` skeleton and an end-to-end **smoke test** ("null run": empty phantom → white
  image → trivial analysis → bundle written & reread).
- a minimal **napari** GUI shell that opens and displays a bundle.

## Invariant conventions (fixed once, see `CLAUDE.md` and `docs/phase0_*`)

- Arrays indexed `[z, y, x]`, C-contiguous. Right-handed physical frame: x→right, y→up,
  z = depth (≥ 0 at the surface).
- Lengths in **µm**, angles in **radians** internally; degrees only at the GUI boundary.
- Voxel size `(dz, dy, dx)` stored explicitly; voxel physical coordinate `(ix·dx, iy·dy, iz·dz)`.
- Azimuth `φ ∈ [0, π)` (in the x,y plane, +x→+y), elevation `θ ∈ [−π/2, π/2]`,
  director `n = (cosθ·cosφ, cosθ·sinφ, sinθ)`. **Fiber axis = MINOR eigenvector of the
  structure tensor.**
- Axial orientations (period π) handled via the **doubled angle** `(cos 2φ, sin 2φ)`.
- Reproducibility: everything regenerable from `{config + seed + code version}`.

## Install (development)

The project virtual environment lives outside the repo at
`C:\env_python\env_FibreMap` (Python ≥ 3.11).

```bash
python -m pip install -e ".[dev]"      # core + test tooling
python -m pip install -e ".[dev,gui]"  # add the napari GUI shell
```

## Commands

```bash
pytest                                 # run the test suite
collagen-shg-gui path/to/dataset.bundle   # launch the napari shell on a bundle
ruff check src tests                   # lint
```

## Layout

```
src/collagen_shg/
  representations/   phantom, image_bundle, conventions, io   (Phase 0 — implemented)
  config/            models, loader, seeds                    (Phase 0 — implemented)
  metrics/           structure_tensor, order, correlation,    (Livrable 1 — implemented)
                     fourier, texture, fibers, defects
  structure_generator/  imaging/                               (Livrable 2 — implemented)
  refinement/                                                  (Tier 2 / Phase 3 — stub)
  analysis_resolved/  pshg/                                   (Livrables 3–4 — stubs)
  validation/        closed loop (generate→image→analyze) + null run
  gui/               napari shell
configs/   tissues/ microscopes/ runs/   (YAML presets)
docs/      phase0 / livrable1 / spec     (design references — authoritative)
```