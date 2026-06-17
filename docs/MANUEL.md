# Manuel d'utilisation — collagen-shg (FibreMap)

Guide pratique pour tester **intégralement** le projet : installation, tests, scripts d'exemple,
interface graphique, API Python (génération, imagerie, métriques, analyse, boucle fermée),
configuration et paramètres.

État implémenté : **Phase 0** (socle) + **Livrable 1** (métriques A–G) + **Livrable 2**
(générateurs structure/image + boucle fermée) + **Livrable 3** (analyse tissus résolus).
P-SHG (Livrable 4) et raffinement appris (Tier 2) sont des stubs.

---

## 0. Conventions de ce manuel

- **Dépôt** : `C:\LP2N\Software_FibreMap\FibreMap` — lancez les commandes **depuis ce dossier**.
- **Python du venv** : `C:\env_python\env_FibreMap\Scripts\python.exe`. Dans la suite je l'abrège
  en `python` ; si le venv n'est pas activé, remplacez `python` par le chemin complet ci-dessus.
- Toutes les longueurs sont en **µm**, les angles en **radians en interne** (degrés seulement à
  la frontière config/GUI). Tableaux indexés **`[z, y, x]`**.

---

## 1. Installation / activation

Le venv existe déjà (`C:\env_python\env_FibreMap`, Python 3.13) et le paquet est installé en
mode editable. Pour (ré)installer ou repartir de zéro :

```powershell
# Activer le venv (PowerShell)
C:\env_python\env_FibreMap\Scripts\Activate.ps1

# Depuis le dépôt
cd C:\LP2N\Software_FibreMap\FibreMap

# Cœur + outils de test
python -m pip install -e ".[dev]"

# Avec l'interface graphique napari (gros téléchargement)
python -m pip install -e ".[dev,gui]"
```

Dépendances cœur : pydantic v2, numpy, scipy, scikit-image, zarr, tifffile, pyarrow, pyyaml.
Optionnel : napari (`gui`), pytest/hypothesis/ruff (`dev`).

---

## 2. Lancer les tests (vérification rapide que tout marche)

```powershell
cd C:\LP2N\Software_FibreMap\FibreMap

python -m pytest                       # toute la suite (~130 tests, ~30 s)
python -m pytest -q tests/test_metrics_structure_tensor.py   # un fichier
python -m pytest -k closed_loop        # par mot-clé
python -m ruff check src tests         # lint
```

Découpage des tests par étape :

| Fichier | Couvre |
|---|---|
| `test_conventions.py`, `test_schema.py`, `test_io_roundtrip.py` | Phase 0 (socle, E/S) |
| `test_config.py`, `test_seeds.py`, `test_smoke_e2e.py` | Phase 0 (config, graines, run nul) |
| `test_metrics_*.py` | Livrable 1 (familles A–G) |
| `test_structure_generator.py`, `test_imaging_*.py` | Livrable 2 (générateurs) |
| `test_validation_closed_loop.py` | Livrable 2 (boucle fermée) |
| `test_analysis_*.py` | Livrable 3 (analyse résolue) |
| `test_gui_*.py` | GUI (logique pure, sans napari) |

---

## 3. Scripts d'exemple (le plus simple pour « tout tester »)

Trois scripts prêts à l'emploi dans `examples/`. Lancez-les depuis le dépôt :

```powershell
cd C:\LP2N\Software_FibreMap\FibreMap

python examples/metrics_demo.py        # Livrable 1 : métriques sur motifs synthétiques
python examples/coherent_fb_demo.py    # Livrable 2 : rapport avant/arrière SHG cohérent
python examples/quickstart.py          # bout-en-bout : génère -> image -> sauve -> analyse
```

### 3.1 `quickstart.py` — la boucle fermée de bout en bout

C'est **le** test intégral : il génère un phantom à vérité-terrain connue, l'image (Tier 1),
écrit/relit un *bundle*, puis l'analyse et compare le mesuré à la vérité-terrain.

```powershell
python examples/quickstart.py                          # config rapide (16x128x128, ~secondes)
python examples/quickstart.py configs/runs/demo_tendon.yaml   # taille réelle (64x512x512, lent)
```

Sortie attendue (config `demo_small`) :

```
[1] generated phantom: 64 fibrils
    ground truth: S2=0.968 S3=0.976 mean_phi=90.9 deg
[2] image: shape=(16, 128, 128) dtype=float32 mean=428.7 max=1660.9
[3] bundle written to: ...\datasets\demo_small.bundle
    reload image identical: True
[4] measured vs ground truth:
    S2        measured=  0.685  truth=  0.968  bias=-0.283
    S3        measured=  0.581  truth=  0.976  bias=-0.395
    mean_phi  measured= 91.975 deg  truth= 90.918 deg  bias=+1.057 deg
```

> **Lecture** : l'orientation moyenne est retrouvée à ~1°. Les paramètres d'ordre mesurés sont
> **inférieurs** à la vérité (biais négatif) : c'est l'effet réel de la PSF et du bruit, et c'est
> précisément ce que la boucle fermée sert à **quantifier**.

---

## 4. Interface graphique (napari)

Nécessite l'extra `gui` installé (`pip install -e ".[dev,gui]"`) et un affichage.

### 4.1 Application interactive (le cas principal)

Lancez **sans argument** → une appli à **3 onglets** s'ouvre dans napari (panneau de droite) :

```powershell
collagen-shg-gui
```

- **Onglet 1 · Structure** — réglez la ROI (`Z`, `Y`, `X` voxels ; `voxel_*_um`), le nombre de
  fibrilles (`n_fibrils`), le `diameter_um` + `dispersion`, le crimp (`crimp_amplitude_um`,
  `crimp_period_um`), l'organisation (`mean_phi_deg`, `kappa`, `xi_um`) et la `seed`.
  Cliquez **« Générer la structure »** → s'affichent le volume de densité, l'orientation (RVB :
  teinte = azimut) et le **squelette** des fibrilles ; la vérité-terrain (S2/S3/φ) est affichée
  en notification.
- **Onglet 2 · Imagerie** — réglez le microscope (`NA`, `wavelength_nm`, `detection`,
  `attenuation_length_um`, `photons_peak`, `read_noise_e`, `seed`). Cliquez **« Générer
  l'image »** → image SHG incohérente (modèle scalaire). La case **`realiste_tier2`** applique
  une étape de raffinement *placeholder* (le vrai modèle, entraîné sur données réelles, viendra).
- **Onglet 3 · Analyse** — **« Analyser l'image courante »** extrait l'organisation de l'image
  générée (cartes orientation/cohérence + S2/S3/ξ/défauts en notification). **« Charger &
  analyser »** ouvre un sélecteur de fichier pour une **image réelle** (OME-TIFF) ou un bundle, et
  l'analyse de la même façon (les images 2D passent par les métriques 2D).

> Astuce : le volume est 3D — utilisez le **curseur en bas** pour parcourir les plans z, et l'œil
> 👁 à gauche de chaque couche pour l'afficher/masquer.

### 4.2 Modes rapides (optionnels)

```powershell
# Visualiser un bundle déjà écrit (ou un OME-TIFF), sans panneau de contrôle
collagen-shg-gui datasets/demo_small.bundle
collagen-shg-gui mon_image.ome.tif

# Générer un bundle depuis un fichier de config ET l'afficher
collagen-shg-gui --generate configs/runs/demo_small.yaml

collagen-shg-gui --help
```

> Si `collagen-shg-gui` n'est pas reconnu (venv non activé) :
> `C:\env_python\env_FibreMap\Scripts\collagen-shg-gui.exe`

---

## 5. API Python par capacité

Toutes les briques sont scriptables. Schéma général : **config → graines → générateur → imageur
→ (E/S) → analyse → comparaison**.

### 5.1 Charger une configuration

```python
from collagen_shg.config import load_config, load_config_dict
cfg = load_config("configs/runs/demo_small.yaml")     # YAML + résolution des presets
cfg.run.seed, cfg.volume.shape_zyx, cfg.structure.orientation.kappa
```

### 5.2 Graines reproductibles

```python
from collagen_shg.config.seeds import SeedManager
seeds = SeedManager(cfg.run.seed)
rng_struct = seeds.generator("structure")   # flux indépendant nommé
rng_noise  = seeds.generator("noise")
seeds.provenance()                           # {rng, master_seed, children}
```

### 5.3 Tier 0 — générer une structure (phantom)

```python
from collagen_shg.structure_generator import ProceduralStructureGenerator
gen = ProceduralStructureGenerator(
    cfg.volume.shape_zyx, cfg.volume.voxel_size_zyx_um,
    n_fibrils=200,        # défaut: ~ (X*Y)//256
    samples_per_um=2.0,   # densité d'échantillonnage des centerlines
)
phantom = gen.generate(cfg.structure, rng_struct)
phantom.ground_truth.global_.S2          # vérité-terrain mesurée sur la structure
phantom.fields.director.shape            # (3, Z, Y, X)
len(phantom.geometry)                    # liste de Fibril (centerlines en µm)
```

### 5.4 Tier 1 — imagerie incohérente

```python
from collagen_shg.imaging import IncoherentImager
bundle = IncoherentImager().render(phantom, cfg.microscope, cfg.degradation, rng_noise)
# signal sans bruit (pour debug / tests) :
signal = IncoherentImager().signal(phantom, cfg.microscope, cfg.degradation)
# image sans bruit :
clean  = IncoherentImager().render(phantom, cfg.microscope, cfg.degradation, rng_noise,
                                   add_noise=False)
```

### 5.5 Tier 3 — imagerie cohérente (avant/arrière)

```python
from collagen_shg.imaging import CoherentImager
cf = CoherentImager().fields(phantom, cfg.microscope)   # forward, backward, fb_ratio
bundle2d = CoherentImager().render(phantom, cfg.microscope, cfg.degradation, rng_noise)
# -> image 2D détectée [1, Y, X] (projection cohérente ; pas de GT volumétrique attachée)
```

### 5.6 E/S — sauver / relire un bundle ; ingérer une image réelle

```python
from collagen_shg.representations import write_bundle, read_bundle, read_ome_tiff
write_bundle(bundle, "datasets/run.bundle", overwrite=True)
b = read_bundle("datasets/run.bundle")           # round-trip bit-à-bit
real = read_ome_tiff("image.ome.tif", voxel_size_zyx=(0.5, 0.2, 0.2))   # kind="real"
```

Contenu d'un `*.bundle/` : `image.zarr/`, `ground_truth/{fields.zarr, geometry.parquet,
phantom_meta.json, organization.json}`, `metadata.json`, `config.yaml`, `provenance.json`.

### 5.7 Métriques (Livrable 1, familles A–G)

```python
from collagen_shg import metrics
st  = metrics.structure_tensor_2d(image2d, sigma=1.0, rho=4.0)      # A (2D)
st3 = metrics.structure_tensor_3d(volume, sigma=1.0, rho=2.0)       # A (3D)
op  = metrics.order_parameter_2d(st.orientation, st.coherence)      # B : S2, theta_bar, kappa
ot  = metrics.order_tensor_3d(st3.director, weights=st3.fa)         # B : S3, directeur, Q
cc  = metrics.orientation_correlation(field, max_r=16)              # C : C(r), xi, plateau
ps  = metrics.power_spectrum_orientation(image2d)                   # D : A(phi), orientation, spacing
glcm = metrics.glcm_features(image2d)                              # E
lbp  = metrics.lbp_histogram(image2d, P=8, R=1.0)                  # E
gab  = metrics.gabor_energy(image2d)                              # E
fm   = metrics.fiber_metrics([fibril_or_centerline, ...])          # F
lp   = metrics.persistence_length(centerline_Nx3)                  # F
dd   = metrics.defect_density(theta2d)                             # G
```

### 5.8 Boucle fermée (générer → imager → analyser → comparer)

```python
from collagen_shg.validation import run_closed_loop
rep = run_closed_loop(cfg, sigma=1.0, rho=2.0, n_fibrils=300)
rep.measured       # {S2, S3, mean_phi, kappa, ...}
rep.ground_truth   # vérité-terrain
rep.bias           # mesuré - vérité (biais axial-circulaire pour mean_phi)
```

### 5.9 Analyse des tissus résolus (Livrable 3)

```python
from collagen_shg.analysis_resolved import ResolvedAnalyzer
analyzer = ResolvedAnalyzer(
    sigma=1.0, rhos=(1.0, 2.0, 4.0),   # tenseur multi-échelle
    flat_field=True, denoise_sigma=0.0, subtract_bg=False,   # prétraitement
    max_r=16,                          # portée de la corrélation ξ
    bootstrap=True, n_boot=200,        # IC bootstrap sur S2/S3
)
res = analyzer.analyze_bundle(b)       # ou analyzer.analyze(volume, voxel_size_zyx)
res.descriptors        # OrganizationDescriptors (S2, S3, xi_um, defect_density, mean_phi, fa_mean)
res.orientation        # carte azimut [Z,Y,X]
res.coherence          # carte FA [Z,Y,X]
res.descriptor_vector  # vecteur de features (longueur fixe)
res.ci                 # {"S2": (lo, hi), "S3": (lo, hi)} si bootstrap=True
```

### 5.10 Comparaison inter-tissus (PCA + classifieur)

```python
from collagen_shg.analysis_resolved import (
    feature_matrix, standardize, PCA, NearestCentroidClassifier)
X = feature_matrix([res_a.descriptor_vector, res_b.descriptor_vector, ...])
Xs, mean, std = standardize(X)
pcs = PCA(n_components=2).fit_transform(Xs)
clf = NearestCentroidClassifier().fit(Xs, labels)   # labels: ["tendon", "skin", ...]
clf.predict(Xs)
```

---

## 6. Configuration YAML — structure complète

Un run est entièrement décrit par un YAML (validé en objets typés). Les `preset` héritent d'un
fragment dans `configs/tissues/` ou `configs/microscopes/`, surchargé par `overrides`.

```yaml
run:
  name: demo_small
  seed: 20260616                 # graine maître (reproductibilité)
  output: datasets/demo_small.bundle

volume:
  shape_zyx: [16, 128, 128]      # [Z, Y, X] en voxels
  voxel_size_zyx_um: [0.5, 0.2, 0.2]   # (dz, dy, dx) en µm

structure:
  preset: tendon                 # configs/tissues/tendon.yaml
  overrides:
    orientation: { mean_phi_deg: 90, kappa: 20, xi_um: 40 }
    fibril:
      diameter_um: { mean: 1.5, dispersion: 0.3 }
      crimp: { amplitude_um: 2.0, period_um: 25 }

microscope:
  preset: default                # configs/microscopes/default.yaml
  overrides:
    mode: incoherent             # incoherent | coherent
    NA: 0.95
    wavelength_nm: 900
    detection: backward          # backward | forward
    pixel_size_um: 0.2

degradation:
  depth: { attenuation_length_um: 80 }
  noise: { photons_peak: 2000, read_noise_e: 2.0 }
```

Presets fournis : tissus `tendon`, `skin`, `cornea` ; microscope `default`. Runs : `demo_small`
(rapide) et `demo_tendon` (taille réelle).

---

## 7. Référence des paramètres

| Bloc | Paramètre | Rôle / effet |
|---|---|---|
| `volume` | `shape_zyx` | dimensions du volume `[Z, Y, X]` (voxels) |
| | `voxel_size_zyx_um` | taille de voxel `(dz, dy, dx)` en µm |
| `structure.orientation` | `mean_phi_deg` | azimut moyen des fibres (degrés, frontière) |
| | `kappa` | concentration (↑ = plus aligné ; ~0 = isotrope) |
| | `xi_um` | longueur de corrélation cible (nominal pour l'instant) |
| | `elevation_sigma` | dispersion d'élévation (optionnel, défaut 0 = in-plane) |
| `structure.fibril` | `diameter_um.mean/dispersion` | diamètre des fibrilles (µm) |
| | `crimp.amplitude_um/period_um` | ondulation sinusoïdale (signature tendon) |
| `microscope` | `mode` | `incoherent` (Tier 1) ou `coherent` (Tier 3) |
| | `NA`, `wavelength_nm` | déterminent la PSF (FWHM latérale ≈ 0.51·λ/NA) |
| | `detection` | `backward` (épi, ×2 atténuation) ou `forward` |
| | `pixel_size_um` | échantillonnage latéral (métadonnée) |
| `degradation.depth` | `attenuation_length_um` | atténuation Beer–Lambert en profondeur |
| `degradation.noise` | `photons_peak` | photons au pic → bruit de Poisson |
| | `read_noise_e` | bruit de lecture gaussien (e⁻) |

Paramètres « code » non dans le YAML :

| Objet | Paramètre | Rôle |
|---|---|---|
| `ProceduralStructureGenerator` | `n_fibrils`, `samples_per_um` | nb de fibrilles, densité d'échantillonnage |
| `IncoherentImager.render` | `add_noise` | activer/désactiver le bruit |
| `CoherentImager` | `n`, `dispersion_dk` | indice du milieu, désaccord avant |
| `ResolvedAnalyzer` | `sigma`, `rhos` | échelles bruit/intégration du tenseur |
| | `flat_field`, `denoise_sigma`, `subtract_bg` | prétraitement |
| | `max_r`, `bootstrap`, `n_boot` | portée ξ, IC bootstrap |

---

## 8. Reproductibilité

Tout artefact se régénère à partir de `{config + graine + version du code}`. Même `seed` ⇒
sorties **numériquement identiques** (Tiers 0–1, déterministes). La graine maître est dans
`run.seed` ; les flux enfants (`structure`, `noise`, …) sont dérivés par `SeedManager` et
consignés dans la provenance du bundle (`provenance.json`).

---

## 9. Limites connues (état actuel)

- `ResolvedAnalyzer.analyze` attend un **volume 3D** `[Z, Y, X]` (z-stack). Le chemin 2D pur
  (image à plan unique) n'est pas encore branché — utilisez les métriques 2D directement.
- L'imagerie **cohérente** est un modèle scalaire de **1er ordre** (projection 2D, rapport F/B) ;
  la version vectorielle (Richards–Wolf) + GPU est ultérieure.
- L'**extracteur de fibres appris** (DL) est un stub (`LearnedExtractor`) : il demande une infra
  d'entraînement/GPU. L'analyse d'organisation par champ n'en dépend pas.
- Structure de **domaines / ξ explicite** dans le générateur, **UMAP**, et les **balayages
  quantitatifs** complets (SNR/profondeur/dispersion au-dessus de `run_closed_loop`) sont à venir.
- **P-SHG** (Livrable 4) : non implémenté (stub `pshg`).

---

## 10. Aide-mémoire (commandes essentielles)

```powershell
cd C:\LP2N\Software_FibreMap\FibreMap

python -m pytest                                   # tous les tests
python examples/quickstart.py                      # boucle fermée bout-en-bout
python examples/metrics_demo.py                    # démo métriques
python examples/coherent_fb_demo.py                # démo SHG cohérente
collagen-shg-gui                                   # appli interactive 3 onglets (napari)
collagen-shg-gui --generate configs/runs/demo_small.yaml   # générer + voir (napari)
collagen-shg-gui datasets/demo_small.bundle        # voir un bundle existant
```
