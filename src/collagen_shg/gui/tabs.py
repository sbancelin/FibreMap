"""Tabbed napari application — Structure / Image / Analysis (tabs on top).

The Generation tab is organized in three blocks (the user-facing design):

- **Block 1 — Imaged volume**: a table ``Size (µm)`` / ``Voxel (µm)`` × ``X/Y/Z`` with a derived,
  read-only ``# voxels`` row.
- **Block 2 — Single fibril**: per-fibril morphology — diameter & length (mean + CV), waviness
  (persistence length + crimp), and amount (number of fibrils or volume fraction).
- **Block 3 — Network organization**: the architecture (uniaxial / biaxial / lamellar / arcade /
  tubular / isotropic) and its parameters, plus the biaxial dispersion ``κ∥/κ⊥`` and correlation
  length ξ — the descriptors the Analysis tab aims to recover.

*Generate* builds the **binary** fibril structure (1 = fibril, 0 = empty) and displays it.

Heavy logic is in pure, unit-tested functions; napari / magicgui / Qt are imported lazily.
"""

from typing import Any

import numpy as np

from collagen_shg.gui.interactive import (
    AppState,
    analyze_auto,
    degradation_config_from_params,
    descriptors_summary,
    image_phantom,
    microscope_config_from_params,
    refine_placeholder,
    skeleton_paths,
)
from collagen_shg.gui.orientation import director_to_rgb

__all__ = [
    "voxel_counts",
    "build_volume",
    "arch_params_for",
    "build_structure_config",
    "generate_structure_phantom",
    "run_app",
]

ARCHITECTURE_CHOICES = ("uniaxial", "biaxial", "lamellar", "arcade", "tubular", "isotropic")


# --------------------------------------------------------------------------- pure helpers
def voxel_counts(
    size_xyz: tuple[float, float, float], voxel_xyz: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Derived voxel counts ``# = round(Size / Voxel)`` per axis (≥ 1)."""
    return tuple(
        max(1, int(round(s / v))) if v > 0 else 1
        for s, v in zip(size_xyz, voxel_xyz, strict=True)
    )


def build_volume(
    size_xyz: tuple[float, float, float], voxel_xyz: tuple[float, float, float]
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """GUI (X, Y, Z) size/voxel → array ``(shape_zyx, voxel_size_zyx)``."""
    nx, ny, nz = voxel_counts(size_xyz, voxel_xyz)
    vx, vy, vz = voxel_xyz
    return (nz, ny, nx), (vz, vy, vx)


def arch_params_for(architecture: str, values: dict[str, Any]) -> dict[str, Any]:
    """Select the parameters relevant to ``architecture`` from a flat GUI value dict."""
    a = architecture.lower()
    if a in ("uniaxial", "isotropic"):
        return {"mean_phi_deg": values.get("mean_phi_deg", 0.0),
                "mean_theta_deg": values.get("mean_theta_deg", 0.0)}
    if a == "biaxial":
        return {"phi_a_deg": values.get("phi_a_deg", 0.0),
                "phi_b_deg": values.get("phi_b_deg", 90.0),
                "mix": values.get("mix", 0.5)}
    if a == "lamellar":
        return {"mean_phi_deg": values.get("mean_phi_deg", 0.0),
                "lamella_thickness_um": values.get("lamella_thickness_um", 2.0),
                "lamella_dphi_deg": values.get("lamella_dphi_deg", 90.0)}
    if a == "arcade":
        return {"mean_phi_deg": values.get("mean_phi_deg", 0.0),
                "theta_deep_deg": values.get("theta_deep_deg", 90.0),
                "theta_surface_deg": values.get("theta_surface_deg", 0.0)}
    # tubular
    return {"helix_beta_deg": values.get("helix_beta_deg", 0.0),
            "crossed": bool(values.get("crossed", False))}


def build_structure_config(
    architecture: str,
    arch_params: dict[str, Any],
    *,
    kappa_par: float,
    kappa_perp: float,
    xi_um: float,
    diameter_um: float,
    diameter_cv: float,
    length_um: float,
    length_cv: float,
    persistence_um: float,
    crimp_amplitude_um: float,
    crimp_period_um: float,
    volume_fraction: float | None = None,
):
    """Assemble a :class:`StructureConfig` from the GUI blocks (architecture-aware)."""
    from collagen_shg.config.models import StructureConfig

    data: dict[str, Any] = {
        "architecture": {"type": architecture, **arch_params},
        "orientation": {
            "mean_phi_deg": arch_params.get("mean_phi_deg", 0.0),
            "kappa_par": kappa_par,
            "kappa_perp": kappa_perp,
            "xi_um": xi_um,
        },
        "fibril": {
            "diameter_um": {"mean": diameter_um, "dispersion": diameter_cv},
            "length_um": length_um,
            "length_cv": length_cv,
            "persistence_um": persistence_um,
            "crimp": {"amplitude_um": crimp_amplitude_um, "period_um": crimp_period_um},
        },
    }
    if volume_fraction is not None:
        data["volume_fraction"] = volume_fraction
    return StructureConfig.model_validate(data)


def generate_structure_phantom(
    size_xyz, voxel_xyz, architecture, arch_params, *, seed, n_fibrils=None, **fibril_org,
):
    """Build a binary-tube phantom from GUI parameters (deterministic for ``seed``)."""
    from collagen_shg.structure_generator.generator import ProceduralStructureGenerator

    shape_zyx, voxel_zyx = build_volume(size_xyz, voxel_xyz)
    config = build_structure_config(architecture, arch_params, **fibril_org)
    generator = ProceduralStructureGenerator(shape_zyx, voxel_zyx, n_fibrils=n_fibrils)
    return generator.generate(config, np.random.default_rng(int(seed)))


# --------------------------------------------------------------------------- display
def _show_structure(state: AppState) -> None:
    viewer, phantom = state.viewer, state.phantom
    scale = phantom.meta.voxel_size_zyx
    _set_layer(viewer, "fibrils (binary)", np.asarray(phantom.fields.density), "image",
               scale=scale, colormap="gray", blending="additive")
    rgb = director_to_rgb(np.asarray(phantom.fields.director), np.asarray(phantom.fields.order_S))
    _set_layer(viewer, "orientation (GT)", rgb, "image", scale=scale, rgb=True, visible=False)
    paths = skeleton_paths(phantom)
    if "skeleton" in viewer.layers:
        del viewer.layers["skeleton"]
    if paths:
        viewer.add_shapes(paths, shape_type="path", name="skeleton", scale=scale,
                          edge_color="cyan", edge_width=0.3, blending="translucent")


def _show_image(state: AppState) -> None:
    scale = state.bundle.metadata.voxel_size_zyx
    _set_layer(state.viewer, "image", np.asarray(state.bundle.image), "image",
               scale=scale, colormap="gray")


def _show_analysis(state: AppState, result: dict[str, Any]) -> None:
    scale = state.voxel_size
    _set_layer(state.viewer, "orientation (measured)", np.asarray(result["orientation"]),
               "image", scale=scale, colormap="hsv")
    _set_layer(state.viewer, "coherence", np.asarray(result["coherence"]), "image",
               scale=scale, colormap="inferno", visible=False)


def _set_layer(viewer: Any, name: str, data: np.ndarray, kind: str, **kwargs: Any) -> None:
    if name in viewer.layers:
        del viewer.layers[name]
    getattr(viewer, f"add_{kind}")(data, name=name, **kwargs)


# --------------------------------------------------------------------------- tab widgets
def _structure_tab(state: AppState):
    from magicgui.widgets import (
        CheckBox,
        ComboBox,
        Container,
        FloatSpinBox,
        Label,
        PushButton,
        SpinBox,
        Table,
    )
    from napari.utils.notifications import show_info

    volume = Table(
        value={
            "data": [[20.0, 20.0, 10.0], [0.2, 0.2, 0.5], [100, 100, 20]],
            "index": ["Size (µm)", "Voxel (µm)", "# voxels"],
            "columns": ["X", "Y", "Z"],
        }
    )

    def _refresh_counts(*_: Any) -> None:
        data = [list(row) for row in volume.data]
        size = [float(v) for v in data[0]]
        vox = [float(v) for v in data[1]]
        counts = voxel_counts(tuple(size), tuple(vox))
        if [int(c) for c in data[2]] != list(counts):
            volume.value = {
                "data": [data[0], data[1], list(counts)],
                "index": ["Size (µm)", "Voxel (µm)", "# voxels"],
                "columns": ["X", "Y", "Z"],
            }

    volume.changed.connect(_refresh_counts)

    # Block 2 — single fibril
    n_fibrils = SpinBox(value=150, min=1, max=20000, label="Number of fibrils")
    use_fraction = CheckBox(value=False, label="Use volume fraction instead")
    volume_fraction = FloatSpinBox(value=0.1, min=0.0, max=1.0, step=0.01, label="Volume fraction")
    diameter_um = FloatSpinBox(value=1.0, min=0.02, max=20.0, step=0.1, label="Diameter mean (µm)")
    diameter_cv = FloatSpinBox(value=0.3, min=0.0, max=2.0, step=0.05, label="Diameter CV")
    length_um = FloatSpinBox(value=0.0, min=0.0, max=2000.0, step=1.0,
                             label="Length mean (µm, 0=span)")
    length_cv = FloatSpinBox(value=0.0, min=0.0, max=2.0, step=0.05, label="Length CV")
    persistence_um = FloatSpinBox(value=1e6, min=1.0, max=1e6, step=10.0,
                                  label="Persistence Lp (µm)")
    crimp_amplitude_um = FloatSpinBox(value=0.0, min=0.0, max=20.0, step=0.1,
                                      label="Crimp amplitude (µm)")
    crimp_period_um = FloatSpinBox(value=0.0, min=0.0, max=200.0, step=1.0,
                                   label="Crimp period (µm)")

    # Block 3 — organization
    architecture = ComboBox(choices=ARCHITECTURE_CHOICES, value="uniaxial", label="Architecture")
    mean_phi_deg = FloatSpinBox(value=90.0, min=0.0, max=180.0, label="Mean azimuth φ₀ (°)")
    mean_theta_deg = FloatSpinBox(value=0.0, min=-90.0, max=90.0, label="Mean elevation θ₀ (°)")
    phi_a_deg = FloatSpinBox(value=0.0, min=0.0, max=180.0, label="Axis A φ (°)")
    phi_b_deg = FloatSpinBox(value=90.0, min=0.0, max=180.0, label="Axis B φ (°)")
    mix = FloatSpinBox(value=0.5, min=0.0, max=1.0, step=0.05, label="A/B mix")
    lamella_thickness_um = FloatSpinBox(value=2.0, min=0.1, max=50.0,
                                        label="Lamella thickness (µm)")
    lamella_dphi_deg = FloatSpinBox(value=90.0, min=0.0, max=180.0, label="Lamella Δφ (°)")
    theta_deep_deg = FloatSpinBox(value=90.0, min=-90.0, max=90.0, label="Arcade θ deep (°)")
    theta_surface_deg = FloatSpinBox(value=0.0, min=-90.0, max=90.0, label="Arcade θ surface (°)")
    helix_beta_deg = FloatSpinBox(value=0.0, min=-90.0, max=90.0, label="Helix angle β (°)")
    crossed = CheckBox(value=False, label="Crossed ±β")
    kappa_par = FloatSpinBox(value=20.0, min=0.0, max=300.0, label="Dispersion κ∥ (in-plane)")
    kappa_perp = FloatSpinBox(value=20.0, min=0.0, max=300.0, label="Dispersion κ⊥ (out-of-plane)")
    xi_um = FloatSpinBox(value=40.0, min=0.1, max=1000.0, label="Correlation length ξ (µm)")
    seed = SpinBox(value=0, min=0, max=2**31 - 1, label="Seed")

    arch_specific = {
        "uniaxial": [mean_phi_deg, mean_theta_deg],
        "isotropic": [],
        "biaxial": [phi_a_deg, phi_b_deg, mix],
        "lamellar": [mean_phi_deg, lamella_thickness_um, lamella_dphi_deg],
        "arcade": [mean_phi_deg, theta_deep_deg, theta_surface_deg],
        "tubular": [helix_beta_deg, crossed],
    }
    all_arch_widgets = [mean_phi_deg, mean_theta_deg, phi_a_deg, phi_b_deg, mix,
                        lamella_thickness_um, lamella_dphi_deg, theta_deep_deg,
                        theta_surface_deg, helix_beta_deg, crossed]

    def _update_visibility(*_: Any) -> None:
        shown = set(arch_specific.get(str(architecture.value), []))
        for w in all_arch_widgets:
            w.visible = w in shown

    architecture.changed.connect(_update_visibility)
    _update_visibility()

    generate = PushButton(text="Generate structure")
    status = Label(value="")

    def _on_generate(*_: Any) -> None:
        _refresh_counts()
        data = [list(row) for row in volume.data]
        size = tuple(float(v) for v in data[0])
        vox = tuple(float(v) for v in data[1])
        flat = {
            "mean_phi_deg": mean_phi_deg.value, "mean_theta_deg": mean_theta_deg.value,
            "phi_a_deg": phi_a_deg.value, "phi_b_deg": phi_b_deg.value, "mix": mix.value,
            "lamella_thickness_um": lamella_thickness_um.value,
            "lamella_dphi_deg": lamella_dphi_deg.value,
            "theta_deep_deg": theta_deep_deg.value, "theta_surface_deg": theta_surface_deg.value,
            "helix_beta_deg": helix_beta_deg.value, "crossed": crossed.value,
        }
        params = arch_params_for(str(architecture.value), flat)
        phantom = generate_structure_phantom(
            size, vox, str(architecture.value), params, seed=seed.value,
            n_fibrils=None if use_fraction.value else int(n_fibrils.value),
            kappa_par=kappa_par.value, kappa_perp=kappa_perp.value, xi_um=xi_um.value,
            diameter_um=diameter_um.value, diameter_cv=diameter_cv.value,
            length_um=length_um.value, length_cv=length_cv.value,
            persistence_um=persistence_um.value,
            crimp_amplitude_um=crimp_amplitude_um.value, crimp_period_um=crimp_period_um.value,
            volume_fraction=volume_fraction.value if use_fraction.value else None,
        )
        state.phantom = phantom
        state.voxel_size = phantom.meta.voxel_size_zyx
        state.bundle = None
        _show_structure(state)
        gt = phantom.ground_truth.global_
        msg = (f"{len(phantom.geometry)} fibrils | S={gt.S:.2f} biax={gt.biaxiality:.2f} "
               f"φ₀={np.rad2deg(gt.mean_phi):.0f}° φ_v={gt.volume_fraction:.2f}")
        status.value = msg
        show_info("Structure: " + msg)

    generate.changed.connect(_on_generate)

    return Container(
        widgets=[
            Label(value="Block 1 — Imaged volume"), volume,
            Label(value="Block 2 — Single fibril"),
            n_fibrils, use_fraction, volume_fraction,
            diameter_um, diameter_cv, length_um, length_cv,
            persistence_um, crimp_amplitude_um, crimp_period_um,
            Label(value="Block 3 — Network organization"),
            architecture, mean_phi_deg, mean_theta_deg, phi_a_deg, phi_b_deg, mix,
            lamella_thickness_um, lamella_dphi_deg, theta_deep_deg, theta_surface_deg,
            helix_beta_deg, crossed, kappa_par, kappa_perp, xi_um, seed,
            generate, status,
        ],
        labels=True, scrollable=True,
    )


def _image_tab(state: AppState):
    from magicgui.widgets import CheckBox, ComboBox, Container, FloatSpinBox, PushButton, SpinBox
    from napari.utils.notifications import show_info

    NA = FloatSpinBox(value=0.95, min=0.1, max=1.49, label="NA")
    wavelength_nm = FloatSpinBox(value=900.0, min=300.0, max=1300.0, label="Wavelength (nm)")
    detection = ComboBox(choices=("backward", "forward"), value="backward", label="Detection")
    attenuation = FloatSpinBox(value=80.0, min=1.0, max=1000.0, label="Attenuation length (µm)")
    photons = FloatSpinBox(value=2000.0, min=1.0, max=1e5, label="Peak photons")
    read_noise = FloatSpinBox(value=2.0, min=0.0, max=100.0, label="Read noise (e-)")
    realistic = CheckBox(value=False, label="Realistic (Tier 2, placeholder)")
    seed = SpinBox(value=0, min=0, max=2**31 - 1, label="Seed")
    run = PushButton(text="Generate image")

    def _on_run(*_: Any) -> None:
        if state.phantom is None:
            show_info("Generate a structure first (Structure tab).")
            return
        mic = microscope_config_from_params(
            NA=NA.value, wavelength_nm=wavelength_nm.value, detection=str(detection.value)
        )
        deg = degradation_config_from_params(
            attenuation_length_um=attenuation.value, photons_peak=photons.value,
            read_noise_e=read_noise.value,
        )
        bundle = image_phantom(state.phantom, mic, deg, int(seed.value))
        if realistic.value:
            bundle.image = refine_placeholder(bundle.image)
            show_info("Tier 2 refinement: model not trained — placeholder applied.")
        state.bundle = bundle
        _show_image(state)
        show_info(f"Image generated: {bundle.image.shape}")

    run.changed.connect(_on_run)
    return Container(
        widgets=[NA, wavelength_nm, detection, attenuation, photons, read_noise, realistic,
                 seed, run],
        labels=True, scrollable=True,
    )


def _analysis_tab(state: AppState):
    from pathlib import Path

    from magicgui.widgets import Container, FileEdit, Label, PushButton
    from napari.utils.notifications import show_info

    analyze_btn = PushButton(text="Analyze current image")
    file_edit = FileEdit(mode="r", filter="*.tif *.tiff", label="Real image / bundle")
    load_btn = PushButton(text="Load & analyze")
    summary = Label(value="")

    def _do(bundle_image, voxel) -> None:
        result = analyze_auto(np.asarray(bundle_image), voxel)
        state.last_summary = result["summary"]
        _show_analysis(state, result)
        summary.value = result["summary"]
        show_info("Measured organization: " + descriptors_summary(result["descriptors"]))

    def _on_analyze(*_: Any) -> None:
        if state.bundle is None:
            show_info("Generate an image (Image tab) or load one below.")
            return
        _do(state.bundle.image, state.voxel_size)

    def _on_load(*_: Any) -> None:
        from collagen_shg.representations.io import read_bundle, read_ome_tiff

        p = Path(file_edit.value)
        if p.is_dir():
            bundle = read_bundle(p)
        elif p.suffix.lower() in {".tif", ".tiff"}:
            bundle = read_ome_tiff(p)
        else:
            show_info(f"Unsupported path: {p}")
            return
        state.bundle = bundle
        state.voxel_size = bundle.metadata.voxel_size_zyx
        _show_image(state)
        _do(bundle.image, state.voxel_size)

    analyze_btn.changed.connect(_on_analyze)
    load_btn.changed.connect(_on_load)
    return Container(widgets=[analyze_btn, file_edit, load_btn, summary], labels=True)


# --------------------------------------------------------------------------- assembly / launch
def build_tabbed_app(viewer: Any) -> AppState:
    """Attach a single tabbed dock (tabs at the top): Structure / Image / Analysis."""
    from qtpy.QtWidgets import QTabWidget

    state = AppState(viewer=viewer)
    tabs = QTabWidget()
    tabs.addTab(_structure_tab(state).native, "Structure")
    tabs.addTab(_image_tab(state).native, "Image")
    tabs.addTab(_analysis_tab(state).native, "Analysis")
    viewer.window.add_dock_widget(tabs, name="collagen-shg", area="right")
    return state


def run_app(*, show: bool = True, block: bool = True) -> Any:
    """Launch the tabbed interactive application."""
    try:
        import napari
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            'napari is required for the GUI. Install it with:  pip install -e ".[gui]"'
        ) from exc
    viewer = napari.Viewer(show=show, title="collagen-shg")
    build_tabbed_app(viewer)
    if block and show:
        napari.run()
    return viewer