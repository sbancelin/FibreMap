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
)
from collagen_shg.gui.orientation import director_to_rgb

__all__ = [
    "voxel_counts",
    "build_volume",
    "arch_params_for",
    "build_structure_config",
    "generate_structure_phantom",
    "skeleton_volume",
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
    size_xyz, voxel_xyz, architecture, arch_params, *, seed, n_fibrils=None,
    progress=None, **fibril_org,
):
    """Build a binary-tube phantom from GUI parameters (deterministic for ``seed``).

    ``progress`` is an optional ``callable(fraction)`` invoked during fibril growth (0..1).
    """
    from collagen_shg.structure_generator.generator import ProceduralStructureGenerator

    shape_zyx, voxel_zyx = build_volume(size_xyz, voxel_xyz)
    config = build_structure_config(architecture, arch_params, **fibril_org)
    generator = ProceduralStructureGenerator(shape_zyx, voxel_zyx, n_fibrils=n_fibrils)
    return generator.generate(config, np.random.default_rng(int(seed)), progress=progress)


def skeleton_volume(phantom: Any) -> np.ndarray:
    """A thin binary volume marking the fibril centerlines (same coordinates as the tubes).

    A 1-voxel-thick polyline along each fibril; co-centered with ``fibrils (binary)`` by
    construction (identical ``round(point / voxel)`` indexing). Rendered as an image layer (not a
    napari Shapes layer) so it stays aligned and avoids Shapes-related crashes.
    """
    z, y, x = phantom.meta.shape_zyx
    dz, dy, dx = phantom.meta.voxel_size_zyx
    vox = np.array([dx, dy, dz])
    vol = np.zeros((z, y, x), dtype=np.float32)
    for fib in phantom.geometry:
        cl = np.asarray(fib.centerline, dtype=np.float64)
        if cl.shape[0] < 2:
            pts = cl
        else:
            seg = np.diff(cl, axis=0)
            steps = np.maximum(2, (np.linalg.norm(seg / vox, axis=1) * 2).astype(int))
            pts = np.concatenate(
                [cl[k] + seg[k] * np.linspace(0, 1, steps[k])[:, None] for k in range(len(seg))]
            )
        idx = np.round(pts / vox).astype(int)
        ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
        ok = (iz >= 0) & (iz < z) & (iy >= 0) & (iy < y) & (ix >= 0) & (ix < x)
        vol[iz[ok], iy[ok], ix[ok]] = 1.0
    return vol


# --------------------------------------------------------------------------- display
def _update_or_add_image(
    viewer: Any, name: str, data: np.ndarray, *, scale: Any, visible: bool = True, **kw: Any
) -> None:
    """Add an image layer, or update an existing one **in place** (no delete+add churn).

    Reusing the layer avoids the crashes/instability that delete+add of a selected layer can
    cause; falls back to recreating the layer only if an in-place update fails.
    """
    data = np.asarray(data)
    if name in viewer.layers:
        try:
            layer = viewer.layers[name]
            layer.data = data
            layer.scale = scale
            layer.visible = visible
            return
        except Exception:
            del viewer.layers[name]
    viewer.add_image(data, name=name, scale=scale, visible=visible, **kw)


def _show_structure(
    state: AppState, *, fibrils: bool = True, skeleton: bool = True, orientation: bool = False
) -> None:
    viewer, phantom = state.viewer, state.phantom
    scale = phantom.meta.voxel_size_zyx
    _update_or_add_image(viewer, "fibrils (binary)", phantom.fields.density,
                         scale=scale, visible=fibrils, colormap="gray", blending="additive")
    _update_or_add_image(viewer, "skeleton", skeleton_volume(phantom),
                         scale=scale, visible=skeleton, colormap="green", blending="additive")
    rgb = director_to_rgb(np.asarray(phantom.fields.director), np.asarray(phantom.fields.order_S))
    _update_or_add_image(viewer, "orientation (GT)", rgb, scale=scale, visible=orientation,
                         rgb=True)


def _toggle_layer(viewer: Any, name: str, visible: bool) -> None:
    if name in viewer.layers:
        viewer.layers[name].visible = visible


def _show_image(state: AppState) -> None:
    _update_or_add_image(state.viewer, "image", state.bundle.image,
                         scale=state.bundle.metadata.voxel_size_zyx, colormap="gray")


def _show_analysis(state: AppState, result: dict[str, Any]) -> None:
    scale = state.voxel_size
    _update_or_add_image(state.viewer, "orientation (measured)", result["orientation"],
                         scale=scale, colormap="hsv")
    _update_or_add_image(state.viewer, "coherence", result["coherence"],
                         scale=scale, visible=False, colormap="inferno")


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

    # --- Block 1: imaged volume (Size/Voxel x X/Y/Z; # voxels derived, read-only) ---
    volume = Table(
        value={
            "data": [[20.0, 20.0, 10.0], [0.2, 0.2, 0.5], [100, 100, 20]],
            "index": ["Size (µm)", "Voxel (µm)", "# voxels"],
            "columns": ["X", "Y", "Z"],
        }
    )

    def _refresh_counts(*_: Any) -> None:
        data = [list(row) for row in volume.data]
        counts = voxel_counts(
            tuple(float(v) for v in data[0]), tuple(float(v) for v in data[1])
        )
        if [int(c) for c in data[2]] != list(counts):
            volume.value = {
                "data": [data[0], data[1], list(counts)],
                "index": ["Size (µm)", "Voxel (µm)", "# voxels"],
                "columns": ["X", "Y", "Z"],
            }

    volume.changed.connect(_refresh_counts)

    # --- Block 2: single fibril geometry ---
    amount_mode = ComboBox(choices=["Number of fibrils", "Volume fraction"],
                           value="Number of fibrils", label="Amount by")
    n_fibrils = SpinBox(value=40, min=1, max=20000, label="Number of fibrils")
    volume_fraction = FloatSpinBox(value=0.1, min=0.0, max=1.0, step=0.01,
                                   label="Volume fraction φ_v")

    def _amount_visibility(*_: Any) -> None:
        is_number = amount_mode.value == "Number of fibrils"
        n_fibrils.visible = is_number
        volume_fraction.visible = not is_number

    amount_mode.changed.connect(_amount_visibility)
    _amount_visibility()

    # Diameter / length distribution table (CV = coefficient of variation = std / mean).
    morphology = Table(
        value={
            "data": [[1.0, 0.3], [0.0, 0.0]],
            "index": ["Diameter (µm)", "Length (µm)"],
            "columns": ["Mean", "CV"],
        }
    )
    morphology.native.setToolTip("CV = coefficient of variation = std / mean.\n"
                                 "Length mean = 0 means fibrils span the volume.")

    persistence_um = FloatSpinBox(value=1e6, min=1.0, max=1e6, step=10.0,
                                  label="Persistence Lp (µm)")
    crimp_amplitude_um = FloatSpinBox(value=0.0, min=0.0, max=20.0, step=0.1,
                                      label="Amplitude (µm)")
    crimp_period_um = FloatSpinBox(value=0.0, min=0.0, max=200.0, step=1.0, label="Period (µm)")
    crimp_row = Container(widgets=[crimp_amplitude_um, crimp_period_um],
                          layout="horizontal", label="Crimp", labels=True)

    # --- Block 3: network organization ---
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

    # --- Display options (which layers to show after Generate) ---
    show_fibrils = CheckBox(value=True, label="Fibrils (binary)")
    show_skeleton = CheckBox(value=True, label="Skeleton")
    show_orientation = CheckBox(value=False, label="Orientation (GT)")
    show_fibrils.changed.connect(
        lambda *_: _toggle_layer(state.viewer, "fibrils (binary)", show_fibrils.value))
    show_skeleton.changed.connect(
        lambda *_: _toggle_layer(state.viewer, "skeleton", show_skeleton.value))
    show_orientation.changed.connect(
        lambda *_: _toggle_layer(state.viewer, "orientation (GT)", show_orientation.value))
    display_row = Container(widgets=[show_fibrils, show_skeleton, show_orientation],
                            layout="horizontal", label="Display", labels=False)

    generate = PushButton(text="Generate structure")
    status = Label(value="")

    def _collect_params() -> dict[str, Any]:
        """Read every widget on the GUI thread into a plain kwargs dict for the worker."""
        _refresh_counts()
        data = [list(row) for row in volume.data]
        size = tuple(float(v) for v in data[0])
        vox = tuple(float(v) for v in data[1])
        morph = [list(r) for r in morphology.data]
        flat = {
            "mean_phi_deg": mean_phi_deg.value, "mean_theta_deg": mean_theta_deg.value,
            "phi_a_deg": phi_a_deg.value, "phi_b_deg": phi_b_deg.value, "mix": mix.value,
            "lamella_thickness_um": lamella_thickness_um.value,
            "lamella_dphi_deg": lamella_dphi_deg.value,
            "theta_deep_deg": theta_deep_deg.value, "theta_surface_deg": theta_surface_deg.value,
            "helix_beta_deg": helix_beta_deg.value, "crossed": crossed.value,
        }
        params = arch_params_for(str(architecture.value), flat)
        by_number = amount_mode.value == "Number of fibrils"
        return dict(
            size_xyz=size, voxel_xyz=vox, architecture=str(architecture.value),
            arch_params=params, seed=seed.value,
            n_fibrils=int(n_fibrils.value) if by_number else None,
            kappa_par=kappa_par.value, kappa_perp=kappa_perp.value, xi_um=xi_um.value,
            diameter_um=float(morph[0][0]), diameter_cv=float(morph[0][1]),
            length_um=float(morph[1][0]), length_cv=float(morph[1][1]),
            persistence_um=persistence_um.value,
            crimp_amplitude_um=crimp_amplitude_um.value, crimp_period_um=crimp_period_um.value,
            volume_fraction=None if by_number else volume_fraction.value,
        )

    def _on_generate(*_: Any) -> None:
        # Run the (slow) generation off the Qt thread so the GUI stays responsive; the worker's
        # ``returned`` signal fires back on the main thread, where touching napari layers is safe.
        from napari.qt.threading import thread_worker

        kwargs = _collect_params()
        report = state.extra.get("report_progress")  # thread-safe Qt-signal emit (or None)
        bar = state.extra.get("progress")
        generate.enabled = False  # block re-entrant generation (the relaunch crash)
        if bar is not None:
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("Generating… %p%")
        status.value = "Generating…"

        @thread_worker
        def _work() -> Any:
            return generate_structure_phantom(progress=report, **kwargs)

        def _on_done(phantom: Any) -> None:
            state.phantom = phantom
            state.voxel_size = phantom.meta.voxel_size_zyx
            state.bundle = None
            _show_structure(state, fibrils=show_fibrils.value, skeleton=show_skeleton.value,
                            orientation=show_orientation.value)
            gt = phantom.ground_truth.global_
            msg = (f"{len(phantom.geometry)} fibrils | S={gt.S:.2f} biax={gt.biaxiality:.2f} "
                   f"φ₀={np.rad2deg(gt.mean_phi):.0f}° φ_v={gt.volume_fraction:.2f}")
            status.value = msg
            show_info("Structure: " + msg)
            if bar is not None:
                bar.setValue(100)
                bar.setFormat("Done")

        def _on_error(exc: BaseException) -> None:
            import logging

            logging.getLogger("collagen_shg").error(
                "Structure generation failed", exc_info=exc)
            status.value = f"Generation failed: {exc}"
            show_info(f"Generation failed: {exc}")
            if bar is not None:
                bar.setFormat("Error")

        def _on_finish() -> None:
            generate.enabled = True  # re-arm even if generation failed

        worker = _work()
        worker.returned.connect(_on_done)
        worker.errored.connect(_on_error)
        worker.finished.connect(_on_finish)
        worker.start()

    generate.changed.connect(_on_generate)

    # Show all rows of both tables without an inner scrollbar.
    _fit_table(volume, n_rows=3)
    _fit_table(morphology, n_rows=2)

    return Container(
        widgets=[
            Label(value="Imaged volume"), volume,
            Label(value="Single fibril geometry"),
            amount_mode, n_fibrils, volume_fraction,
            morphology, persistence_um, crimp_row,
            Label(value="Network organization"),
            architecture, mean_phi_deg, mean_theta_deg, phi_a_deg, phi_b_deg, mix,
            lamella_thickness_um, lamella_dphi_deg, theta_deep_deg, theta_surface_deg,
            helix_beta_deg, crossed, kappa_par, kappa_perp, xi_um, seed,
            display_row, generate, status,
        ],
        labels=True, scrollable=True,
    )


def _fit_table(table: Any, *, n_rows: int) -> None:
    """Size a magicgui Table so all rows (plus header) are visible without scrolling."""
    from qtpy.QtCore import Qt

    native = table.native
    native.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    native.resizeRowsToContents()
    header = native.horizontalHeader().height()
    row_h = native.rowHeight(0) or 28
    height = header + n_rows * row_h + 2 * native.frameWidth() + 4
    native.setMinimumHeight(int(height))
    native.setMaximumHeight(int(height) + 8)


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
    """Attach a single tabbed dock (tabs at the top): Structure / Image / Analysis.

    A shared progress bar sits at the bottom of the dock; long jobs report into it through a
    thread-safe progress bridge stored on ``state.extra`` so the worker never touches Qt widgets
    directly (cross-thread ``Signal`` delivery is queued onto the GUI thread).
    """
    from qtpy.QtCore import QObject, Signal
    from qtpy.QtWidgets import QProgressBar, QTabWidget, QVBoxLayout, QWidget

    class _ProgressBridge(QObject):
        tick = Signal(float)

    state = AppState(viewer=viewer)

    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(0)
    progress.setTextVisible(True)
    progress.setFormat("Ready")
    bridge = _ProgressBridge()
    bridge.tick.connect(lambda f: progress.setValue(int(f * 100)))
    state.extra["progress"] = progress
    state.extra["progress_bridge"] = bridge  # keep the QObject alive
    state.extra["report_progress"] = bridge.tick.emit  # call from any thread

    tabs = QTabWidget()
    tabs.addTab(_structure_tab(state).native, "Structure")
    tabs.addTab(_image_tab(state).native, "Image")
    tabs.addTab(_analysis_tab(state).native, "Analysis")

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(tabs)
    layout.addWidget(progress)
    viewer.window.add_dock_widget(container, name="collagen-shg", area="right")
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