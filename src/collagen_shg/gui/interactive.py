"""Interactive napari application — Génération / Imagerie / Analyse (Livrable 2-3 GUI).

A three-tab control panel docked in napari:

1. **Structure** — set the ROI (Z/Y/X, voxel µm), fibril count/diameter/crimp and organization
   (mean azimuth, dispersion ``kappa``); click *Générer* → the density volume, the orientation
   (RGB) and the fibril **skeleton** are shown, and the known ground truth is reported.
2. **Imagerie** — set the microscope (NA, λ, detection, depth attenuation, photons, read noise);
   click *Générer l'image* → a scalar incoherent SHG image. An optional *réaliste (Tier 2)* step
   is a labelled placeholder (the learned model is trained on real data later).
3. **Analyse** — analyze the current image, or load a real OME-TIFF / bundle, and read out the
   extracted organization descriptors (orientation/coherence maps + S₂/S₃/ξ/defects).

The heavy logic lives in plain functions (unit-tested without Qt); napari/magicgui are imported
lazily so importing this module never requires the optional ``gui`` extra.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from collagen_shg.config.models import (
    DegradationConfig,
    MicroscopeConfig,
    StructureConfig,
)
from collagen_shg.gui.orientation import director_to_rgb
from collagen_shg.imaging.incoherent import IncoherentImager

__all__ = [
    "AppState",
    "structure_config_from_params",
    "microscope_config_from_params",
    "degradation_config_from_params",
    "generate_structure",
    "image_phantom",
    "skeleton_paths",
    "refine_placeholder",
    "analyze_auto",
    "descriptors_summary",
    "build_app",
    "run_app",
]


# --------------------------------------------------------------------------- application state
@dataclass
class AppState:
    viewer: Any = None
    phantom: Any = None
    bundle: Any = None
    voxel_size: tuple[float, float, float] = (0.5, 0.2, 0.2)
    last_summary: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- pure config helpers
def structure_config_from_params(
    *,
    mean_phi_deg: float,
    kappa: float,
    diameter_um: float,
    dispersion: float,
    crimp_amplitude_um: float,
    crimp_period_um: float,
    xi_um: float = 40.0,
) -> StructureConfig:
    return StructureConfig.model_validate(
        {
            "orientation": {"mean_phi_deg": mean_phi_deg, "kappa": kappa, "xi_um": xi_um},
            "fibril": {
                "diameter_um": {"mean": diameter_um, "dispersion": dispersion},
                "crimp": {"amplitude_um": crimp_amplitude_um, "period_um": crimp_period_um},
            },
        }
    )


def microscope_config_from_params(
    *, NA: float, wavelength_nm: float, detection: str, mode: str = "incoherent"
) -> MicroscopeConfig:
    return MicroscopeConfig(
        mode=mode, NA=NA, wavelength_nm=wavelength_nm, detection=detection, psf_model="gaussian"
    )


def degradation_config_from_params(
    *, attenuation_length_um: float, photons_peak: float, read_noise_e: float
) -> DegradationConfig:
    return DegradationConfig.model_validate(
        {
            "depth": {"attenuation_length_um": attenuation_length_um},
            "noise": {"photons_peak": photons_peak, "read_noise_e": read_noise_e},
        }
    )


# --------------------------------------------------------------------------- pure pipeline helpers
def generate_structure(
    shape_zyx: tuple[int, int, int],
    voxel_size_zyx: tuple[float, float, float],
    *,
    n_fibrils: int,
    seed: int,
    **struct_params: float,
) -> Any:
    """Build a phantom from GUI parameters (deterministic for ``seed``)."""
    from collagen_shg.structure_generator.generator import ProceduralStructureGenerator

    config = structure_config_from_params(**struct_params)
    generator = ProceduralStructureGenerator(shape_zyx, voxel_size_zyx, n_fibrils=n_fibrils)
    return generator.generate(config, np.random.default_rng(seed))


def image_phantom(
    phantom: Any, microscope: MicroscopeConfig, degradation: DegradationConfig, seed: int
) -> Any:
    """Render a scalar incoherent image bundle from a phantom (deterministic for ``seed``)."""
    return IncoherentImager().render(
        phantom, microscope, degradation, np.random.default_rng(seed)
    )


def skeleton_paths(phantom: Any, *, max_fibrils: int = 150) -> list[np.ndarray]:
    """Fibril centerlines as napari ``[z, y, x]`` voxel-index paths (for a Shapes layer)."""
    dz, dy, dx = phantom.meta.voxel_size_zyx
    paths: list[np.ndarray] = []
    for fib in phantom.geometry[:max_fibrils]:
        cl = np.asarray(fib.centerline, dtype=np.float64)  # (N, 3) physical (x, y, z) µm
        paths.append(np.stack([cl[:, 2] / dz, cl[:, 1] / dy, cl[:, 0] / dx], axis=-1))
    return paths


def refine_placeholder(image: np.ndarray) -> np.ndarray:
    """Placeholder for the Tier 2 *réaliste* step (the learned model is trained later).

    Applies a mild, label-only contrast adjustment so the action is visible; it does NOT change
    the organization and is not the trained refinement.
    """
    img = np.asarray(image, dtype=np.float64)
    mx = img.max()
    if mx <= 0:
        return img.astype(np.float32)
    norm = img / mx
    return (np.power(norm, 0.8) * mx).astype(np.float32)


def analyze_auto(image: np.ndarray, voxel_size_zyx: tuple[float, float, float]) -> dict[str, Any]:
    """Analyze a 3D volume (ResolvedAnalyzer) or a 2D / single-plane image (2D metrics).

    Returns ``{orientation, coherence, summary, descriptors, ndim}``.
    """
    img = np.asarray(image, dtype=np.float64)
    is_volume = img.ndim == 3 and img.shape[0] > 2
    if is_volume:
        from collagen_shg.analysis_resolved.analyzer import ResolvedAnalyzer

        result = ResolvedAnalyzer().analyze(img, voxel_size_zyx)
        return {
            "orientation": result.orientation,
            "coherence": result.coherence,
            "descriptors": result.descriptors.as_dict(),
            "summary": descriptors_summary(result.descriptors.as_dict()),
            "ndim": 3,
        }

    plane = img[img.shape[0] // 2] if img.ndim == 3 else img
    return _analyze_2d(plane, voxel_size_zyx)


def _analyze_2d(plane: np.ndarray, voxel_size_zyx: tuple[float, float, float]) -> dict[str, Any]:
    from collagen_shg.metrics.defects import defect_density
    from collagen_shg.metrics.order import order_parameter_2d
    from collagen_shg.metrics.structure_tensor import structure_tensor_2d

    st = structure_tensor_2d(plane, 1.0, 4.0)
    op = order_parameter_2d(st.orientation, st.coherence)
    dd = defect_density(st.orientation)
    desc = {
        "S2": float(op.S2),
        "mean_phi": float(op.theta_bar),
        "kappa": float(op.kappa),
        "defect_density": float(dd.density),
    }
    return {
        "orientation": st.orientation,
        "coherence": st.coherence,
        "descriptors": desc,
        "summary": descriptors_summary(desc),
        "ndim": 2,
    }


def descriptors_summary(desc: dict[str, Any]) -> str:
    """One-line human-readable organization summary from a descriptors dict."""
    parts = []
    if desc.get("S2") is not None:
        parts.append(f"S2={desc['S2']:.3f}")
    if desc.get("S3") is not None:
        parts.append(f"S3={desc['S3']:.3f}")
    if desc.get("mean_phi") is not None:
        parts.append(f"phi={np.rad2deg(desc['mean_phi']):.1f}deg")
    if desc.get("xi_um") is not None and np.isfinite(desc["xi_um"]):
        parts.append(f"xi={desc['xi_um']:.1f}um")
    if desc.get("defect_density") is not None:
        parts.append(f"defects={desc['defect_density']:.2e}")
    return "  ".join(parts)


# --------------------------------------------------------------------------- napari layer helpers
def _set_layer(viewer: Any, name: str, data: np.ndarray, kind: str, **kwargs: Any) -> None:
    if name in viewer.layers:
        del viewer.layers[name]
    getattr(viewer, f"add_{kind}")(data, name=name, **kwargs)


def _show_phantom(state: AppState) -> None:
    viewer, phantom = state.viewer, state.phantom
    scale = phantom.meta.voxel_size_zyx
    fields = phantom.fields
    _set_layer(viewer, "density", np.asarray(fields.density), "image",
               scale=scale, colormap="gray")
    rgb = director_to_rgb(np.asarray(fields.director), np.asarray(fields.order_S))
    _set_layer(viewer, "orientation (GT)", rgb, "image", scale=scale, rgb=True, visible=False)
    paths = skeleton_paths(phantom)
    if "skeleton" in viewer.layers:
        del viewer.layers["skeleton"]
    if paths:
        viewer.add_shapes(paths, shape_type="path", name="skeleton", scale=scale,
                          edge_color="cyan", edge_width=0.3)


def _show_image(state: AppState) -> None:
    viewer = state.viewer
    scale = state.bundle.metadata.voxel_size_zyx
    _set_layer(viewer, "image", np.asarray(state.bundle.image), "image",
               scale=scale, colormap="gray")


def _show_analysis(state: AppState, result: dict[str, Any]) -> None:
    viewer = state.viewer
    scale = state.voxel_size
    _set_layer(viewer, "orientation (measured)", np.asarray(result["orientation"]), "image",
               scale=scale, colormap="hsv", visible=True)
    _set_layer(viewer, "coherence", np.asarray(result["coherence"]), "image",
               scale=scale, colormap="inferno", visible=False)


# --------------------------------------------------------------------------- widget builders
def _structure_widget(state: AppState):
    from magicgui import magicgui
    from napari.utils.notifications import show_info

    @magicgui(
        call_button="Générer la structure",
        Z=dict(min=1, max=512), Y=dict(min=8, max=2048), X=dict(min=8, max=2048),
        kappa=dict(min=0.0, max=200.0),
        n_fibrils=dict(min=1, max=5000),
    )
    def widget(
        Z: int = 16, Y: int = 128, X: int = 128,
        voxel_z_um: float = 0.5, voxel_y_um: float = 0.2, voxel_x_um: float = 0.2,
        n_fibrils: int = 120,
        diameter_um: float = 1.0, dispersion: float = 0.3,
        crimp_amplitude_um: float = 1.0, crimp_period_um: float = 20.0,
        mean_phi_deg: float = 90.0, kappa: float = 20.0,
        xi_um: float = 40.0,
        seed: int = 0,
    ) -> None:
        shape = (int(Z), int(Y), int(X))
        voxel = (float(voxel_z_um), float(voxel_y_um), float(voxel_x_um))
        phantom = generate_structure(
            shape, voxel, n_fibrils=int(n_fibrils), seed=int(seed),
            mean_phi_deg=mean_phi_deg, kappa=kappa, diameter_um=diameter_um,
            dispersion=dispersion, crimp_amplitude_um=crimp_amplitude_um,
            crimp_period_um=crimp_period_um, xi_um=xi_um,
        )
        state.phantom = phantom
        state.voxel_size = voxel
        state.bundle = None
        _show_phantom(state)
        gt = phantom.ground_truth.global_
        show_info(
            f"Structure générée : {len(phantom.geometry)} fibrilles | "
            f"vérité-terrain S2={gt.S2:.2f} S3={gt.S3:.2f} "
            f"phi={np.rad2deg(gt.mean_phi):.0f}°"
        )

    return widget


def _imaging_widget(state: AppState):
    from magicgui import magicgui
    from napari.utils.notifications import show_info

    @magicgui(
        call_button="Générer l'image",
        NA=dict(min=0.1, max=1.49),
    )
    def widget(
        NA: float = 0.95,
        wavelength_nm: float = 900.0,
        detection: Literal["backward", "forward"] = "backward",
        attenuation_length_um: float = 80.0,
        photons_peak: float = 2000.0,
        read_noise_e: float = 2.0,
        realiste_tier2: bool = False,
        seed: int = 0,
    ) -> None:
        if state.phantom is None:
            show_info("Génère d'abord une structure (onglet 1).")
            return
        microscope = microscope_config_from_params(
            NA=NA, wavelength_nm=wavelength_nm, detection=detection
        )
        degradation = degradation_config_from_params(
            attenuation_length_um=attenuation_length_um,
            photons_peak=photons_peak, read_noise_e=read_noise_e,
        )
        bundle = image_phantom(state.phantom, microscope, degradation, int(seed))
        if realiste_tier2:
            bundle.image = refine_placeholder(bundle.image)
            show_info("Raffinement Tier 2 : modèle non entraîné — placeholder appliqué.")
        state.bundle = bundle
        _show_image(state)
        show_info(f"Image générée : {bundle.image.shape}, mode={microscope.mode}")

    return widget


def _analysis_widget(state: AppState):
    from magicgui import magicgui
    from napari.utils.notifications import show_info

    @magicgui(call_button="Analyser l'image courante")
    def analyze_current() -> None:
        if state.bundle is None:
            show_info("Génère une image (onglet 2) ou charge-en une ci-dessous.")
            return
        result = analyze_auto(np.asarray(state.bundle.image), state.voxel_size)
        state.last_summary = result["summary"]
        _show_analysis(state, result)
        show_info("Organisation mesurée : " + result["summary"])

    @magicgui(
        call_button="Charger & analyser",
        path=dict(widget_type="FileEdit", mode="r", filter="*.tif *.tiff"),
    )
    def load_and_analyze(path: Path = Path()) -> None:  # noqa: B008
        from collagen_shg.representations.io import read_bundle, read_ome_tiff

        p = Path(path)
        if p.is_dir():
            bundle = read_bundle(p)
        elif p.suffix.lower() in {".tif", ".tiff"}:
            bundle = read_ome_tiff(p)
        else:
            show_info(f"Format non supporté : {p}")
            return
        state.bundle = bundle
        state.voxel_size = bundle.metadata.voxel_size_zyx
        _show_image(state)
        result = analyze_auto(np.asarray(bundle.image), state.voxel_size)
        state.last_summary = result["summary"]
        _show_analysis(state, result)
        show_info("Image réelle analysée : " + result["summary"])

    from magicgui.widgets import Container

    return Container(widgets=[analyze_current, load_and_analyze], labels=False)


# --------------------------------------------------------------------------- app assembly / launch
def build_app(viewer: Any) -> AppState:
    """Attach the three control panels to a napari viewer as tabbed dock widgets."""
    state = AppState(viewer=viewer)
    add = viewer.window.add_dock_widget
    add(_structure_widget(state), name="1 · Structure", area="right")
    add(_imaging_widget(state), name="2 · Imagerie", area="right", tabify=True)
    add(_analysis_widget(state), name="3 · Analyse", area="right", tabify=True)
    return state


def run_app(*, show: bool = True, block: bool = True) -> Any:
    """Launch the interactive napari application."""
    try:
        import napari
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            'napari is required for the GUI. Install it with:  pip install -e ".[gui]"'
        ) from exc
    viewer = napari.Viewer(show=show, title="collagen-shg — génération / analyse")
    build_app(viewer)
    if block and show:
        napari.run()
    return viewer
