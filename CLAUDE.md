# SHG_analysis — Contexte projet (Claude Code)

## But
Quantifier et comparer l'organisation du collagène dans les tissus conjonctifs à partir d'images SHG (signal *backward* principalement), via quatre briques : un **générateur d'images synthétiques** à vérité-terrain connue, des **métriques d'organisation**, une **analyse d'image** (tissus résolus : peau, tendon, corde vocale, os) et une **analyse P-SHG** (tissus non résolus : cornée, cartilage).

## Principe directeur — boucle fermée (NE PAS PERDRE DE VUE)
Le simulateur produit des images dont l'organisation est **connue** → on valide que les analyseurs la retrouvent (biais/variance en fonction du SNR, de la profondeur, de la dispersion), et les images synthétiques servent d'**annotations gratuites** pour entraîner les extracteurs. Les métriques et le simulateur sont **co-conçus** pour que le simulateur sache générer des images à valeur de métrique imposée.

## Documents de référence — À LIRE AVANT D'IMPLÉMENTER
- `docs/phase0_collagene_SHG.docx` — infrastructure : schéma du **phantom**, formats, conventions, architecture, reproductibilité, **critères d'acceptation**.
- `docs/livrable1_collagene_SHG.docx` — **définitions mathématiques** des métriques + protocole de comparaison + **contrat d'implémentation** du module `metrics`.

Ces specs font foi. Extraire le texte si nécessaire (`python-docx` ou `pandoc`).

## Conventions invariantes (Phase 0) — NE PAS DÉVIER
- Tableaux indexés `[z, y, x]`, C-contigu. Repère physique **droit** : x→droite, y→haut, z = profondeur (≥ 0 à la surface).
- Unités : **longueurs en µm**, **angles en radians** en interne. Degrés uniquement à la frontière GUI.
- Taille de voxel `(dz, dy, dx)` stockée explicitement ; coordonnée physique du voxel `(ix·dx, iy·dy, iz·dz)`.
- Orientation : azimut `φ ∈ [0, π)` (plan x,y, de +x vers +y), élévation `θ ∈ [−π/2, π/2]`, directeur `n = (cosθ·cosφ, cosθ·sinφ, sinθ)`. **L'axe de la fibre = vecteur propre MINEUR (plus petite valeur propre) du tenseur de structure.**
- Orientations axiales (période π) manipulées via l'**angle doublé** `(cos 2φ, sin 2φ)`. En 3D : lois de **Watson/Bingham** (PAS von Mises–Fisher, réservée aux données dirigées via la polarité).
- Polarisation `α ∈ [0, π)` ; pour une fibrille dans le plan, `φ(P-SHG) = φ(tenseur)`.
- Reproductibilité : une **graine maître** par run → graines enfants via `numpy.random.SeedSequence` / `PCG64` ; tout artefact régénérable depuis `{config + graine + version du code}`.

## Architecture — `src/collagen_shg/`
`representations` (Phantom, ImageBundle, conventions, io) · `config` (Pydantic + YAML + presets) · `structure_generator` (Tier 0) · `imaging` (modes incoherent/coherent) · `refinement` (Tier 2) · `metrics` · `analysis_resolved` · `pshg` · `validation` (harnais boucle fermée) · `gui`.

Le **phantom** porte trois choses : la géométrie (liste de fibrilles : centerlines en µm, diamètre, polarité ±1), les champs voxelisés (`director [3,Z,Y,X]`, `order_S`, `density`, `polarity`), et la **vérité-terrain d'organisation** (`S2`, `S3`, `kappa`, `xi_um`, `defect_density`, domaines).

## Format de jeu de données — « bundle » reproductible
`dataset.bundle/` = `image.zarr/` (OME-Zarr) + `ground_truth/` (`fields.zarr/`, `geometry.parquet`) + `metadata.json` + `config.yaml` + `provenance.json`. Les images réelles (OME-TIFF) entrent par le **même** chemin (ground_truth vide/partiel).

## Pile technique
Python ≥ 3.11 · Pydantic v2 · NumPy · Zarr (OME-Zarr/NGFF) · tifffile (OME-TIFF) · PyArrow/Parquet · YAML (pydantic-settings) · `numpy` Generator (PCG64) + SeedSequence · pytest (+ hypothesis) · PyTorch/CuPy (Tiers ultérieurs, **PAS** en Phase 0) · CI GitHub Actions.

## Standards de code
Typage strict + Pydantic pour les schémas ; docstrings concises ; déterminisme garanti pour les Tiers 0–1 ; un test au moins par module ; PRs/commits atomiques par module ; aucune dépendance entre `representations`/`io` et les algorithmes de génération ou d'analyse.

## Ordre de construction
1. **Phase 0** : `representations` + `config` + `conventions` (avec tests sur vecteurs/angles connus) + `io` (avec test **round-trip**) + squelette de dépôt + **smoke test « run nul »** (phantom vide → image blanche → analyse triviale). Vérifier les **critères d'acceptation** de `docs/phase0`.
2. **Livrable 1** : module `metrics` (familles A–G du doc) + tests analytiques (champ uniforme → S = 1 ; isotrope → S = 0 ; sinusoïde → orientation/espacement connus). Le *scoring* complet attend le Livrable 2.
3. **Livrable 2** (générateurs : structure puis imagerie incoherent/coherent), puis **3** (analyse résolue), **4** (P-SHG), puis GUI.

## Commandes
- Tests : `pytest`
- (à compléter au fil de l'implémentation : lint, format, build)
