# Générateur de réseaux de collagène — principe, théorie, paramètres, état de l'art

**Projet FibreMap / `collagen_shg`** · Tier 0 (générateur procédural de structures) · juin 2026

> Rapport scientifique exhaustif. Objectif : documenter le **principe directeur**, le **cadre
> théorique**, le **rôle de chaque paramètre**, et la **comparaison à l'état de l'art** du
> générateur de structures fibrillaires. Le rendu d'images SHG réalistes (Tier 1/2) fait l'objet
> d'un document séparé ; ici on traite uniquement la **génération de la structure 3D** (la
> vérité-terrain géométrique et organisationnelle).

---

## 1. Objectif et principe directeur

### 1.1 La boucle fermée de validation

La faiblesse récurrente de l'analyse d'images de collagène (SHG, histologie) est l'**absence de
vérité-terrain quantitative** : les métriques d'organisation (orientation, alignement, longueur de
corrélation, défauts…) sont validées qualitativement, ou sur peu de données réelles annotées à la
main. Le projet repose sur une **boucle fermée** :

```
        paramètres d'organisation CONNUS
                    │
                    ▼
   ┌────────────────────────────────┐
   │  Générateur (ce document)      │  → structure 3D + vérité-terrain (S, Q, ξ, β…)
   └────────────────────────────────┘
                    │
                    ▼
        modèle de microscope (Tier 1/2)  → image SHG synthétique
                    │
                    ▼
        analyseurs / métriques (Livr. 1, 3)  → organisation MESURÉE
                    │
                    ▼
        comparaison MESURÉ ↔ CONNU  → biais, variance, robustesse
```

Le **générateur est le pivot** : il doit produire des structures dont l'organisation est connue
*exactement et indépendamment* de la chaîne d'analyse, et **couvrir l'ensemble des architectures
collagéniques** observées dans les tissus (tendon, peau, cornée, cartilage, paroi vasculaire, os…).
C'est cette exhaustivité de patterns, associée à une vérité-terrain rigoureuse, qui distingue
l'approche.

### 1.2 Cahier des charges du jeu de paramètres

Les paramètres exposés doivent être :

1. **Indépendants** (orthogonaux) — aucun n'est déductible d'un autre ; ce qui est dérivable est
   calculé, pas saisi (ex. l'espacement inter-fibrille se déduit de la densité et du diamètre) ;
2. **Complets** — suffisants pour décrire intégralement, au 2ⁿᵈ ordre statistique, un réseau axial
   de fibres ;
3. **Mesurables** — chaque paramètre correspond à un descripteur que l'analyse cherchera à
   retrouver (correspondance 1:1 avec les familles de métriques A–G).

---

## 2. Cadre théorique

### 2.1 Représentation : champ de directeur et nature *axiale*

Une fibrille est localement décrite par son **directeur** `n`, vecteur unitaire tangent à l'axe de
la fibre. Avec les conventions invariantes du projet (repère physique droit, x→droite, y→haut,
z = profondeur ; longueurs en µm, angles en radians) :

```
n = (cosθ·cosφ, cosθ·sinφ, sinθ)
```

- **azimut** `φ ∈ [0, π)` mesuré dans le plan (x, y) de +x vers +y ;
- **élévation** `θ ∈ [−π/2, π/2]` entre la fibre et le plan (x, y).

Point théorique central : le collagène fibrillaire est **axial** (apolaire), c.-à-d. `n ≡ −n` — une
fibre n'a pas de sens privilégié. Conséquences :

- l'azimut a une **période π** (et non 2π) ;
- les statistiques circulaires doivent être faites sur l'**angle doublé** `(cos 2φ, sin 2φ)` pour
  éviter la discontinuité 0/π ;
- la distribution d'orientation appropriée est de type **Watson/Bingham** (axiale), **pas** von
  Mises–Fisher (réservée aux données *dirigées*, p. ex. via la polarité SHG).

Le générateur stocke deux représentations co-localisées :
- **géométrique** : liste de fibrilles paramétriques (centerlines `[N×3]` en µm, diamètre,
  polarité ±1, `fiber_id`, `fascicle_id`, `type`) — la « vérité-terrain » d'entraînement ;
- **voxelisée** : champs `director [3,Z,Y,X]`, `density [Z,Y,X]` (binaire 1 = fibrille / 0 = vide),
  `order_S`, `polarity` — pour l'imagerie et l'analyse par champ.

### 2.2 Ordre nématique : paramètre d'ordre scalaire et tenseur d'ordre

L'**organisation** d'un ensemble de directeurs est décrite, au 2ⁿᵈ ordre, par le **tenseur d'ordre
nématique** (de Gennes / Saupe) :

```
Q = ⟨ (3 n⊗n − I) / 2 ⟩
```

(moyenne sur les fibres, pondérable). `Q` est symétrique, de **trace nulle** → **5 degrés de
liberté indépendants** : c'est la description complète au 2ⁿᵈ ordre de la fonction de distribution
d'orientation (ODF). On le diagonalise : valeurs propres `λ₁ ≥ λ₂ ≥ λ₃` (λ₁+λ₂+λ₃ = 0).

- **direction moyenne** = vecteur propre de `λ₁` → (φ₀, θ₀) ;
- **paramètre d'ordre scalaire** `S = λ₁ ∈ [−1/2, 1]` (= S₃ en 3D) :
  `S = 1` alignement parfait, `S = 0` isotrope, `S = −1/2` confinement planaire parfait (girdle) ;
- **biaxialité** `β = λ₂ − λ₃` : distingue une distribution **uniaxiale** (β ≈ 0 : tendon, fibres
  serrées autour d'un axe) d'une distribution **biaxiale / en ceinture** (β > 0 : peau, lamelles —
  étalement anisotrope, p. ex. confiné dans un plan mais dispersé dans ce plan).

En 2D (projeté dans le plan), on utilise le paramètre d'ordre du **vecteur résultant moyen de
l'angle doublé** :

```
S₂ = | ⟨ e^{i 2φ} ⟩ |  ∈ [0, 1]
```

`S₂ = 1` aligné, `S₂ = 0` isotrope. Un ajustement von Mises (axial) sur 2φ donne la concentration
`κ`, reliée à `S₂` par le rapport de fonctions de Bessel `I₁/I₀`.

**Pourquoi 5 ddl suffisent (et pourquoi pas plus).** Le tenseur `Q` capture l'orientation moyenne
(2), la force d'alignement (1) et la biaxialité + son orientation (2). Les modes d'ordre supérieur
(distributions multimodales franches, p. ex. deux axes distincts à 60°) nécessiteraient l'ODF
complète (harmoniques sphériques d'ordre 4+). Le générateur les produit néanmoins via des
**mélanges de populations** (cf. §3.1, architecture *biaxiale*), mais la vérité-terrain scalaire se
résume au tenseur `Q` (standard de la communauté cristaux liquides et de l'analyse SHG).

### 2.3 Distributions d'orientation sur la sphère

| Distribution | Densité ∝ | Nature | Usage |
|---|---|---|---|
| von Mises–Fisher (vMF) | `exp(κ μ·n)` | **dirigée** (polaire) | données polarisées/polarité, **pas** l'axe nématique |
| **Watson** | `exp(κ (μ·n)²)` | **axiale, axisymétrique** | dispersion isotrope autour d'un axe ; `κ>0` cluster, `κ<0` girdle |
| **Bingham** | `exp(nᵀ B n)` | **axiale, biaxiale** | dispersion **anisotrope** (≠ dans deux directions ⟂) |

Le générateur échantillonne une loi de type **Bingham** (Watson anisotrope) paramétrée par deux
concentrations `κ∥` (dans le plan d'alignement) et `κ⊥` (hors plan) — voir §3.2. C'est le bon
cadre pour reproduire : tendon (κ∥, κ⊥ élevés → cluster serré), peau en ceinture de Langer
(κ⊥ élevé, κ∥ faible → girdle planaire dispersé), isotrope (κ → 0).

### 2.4 Structure spatiale : longueur de corrélation ξ

Le paramètre d'ordre `S` est **global** : il ne distingue pas un tissu *localement aligné mais
globalement isotrope* (derme : petits domaines, orientations de domaines aléatoires) d'un tissu
*globalement aligné* (tendon). Cette information est portée par la **fonction de corrélation
d'orientation à deux points** :

```
C(r) = ⟨ cos 2(φ(x) − φ(x+r)) ⟩   (2D, angle doublé)
C(r) = ⟨ P₂( n(x)·n(x+r) ) ⟩       (3D, polynôme de Legendre)
```

`C(r)` décroît d'un plateau (= ordre global, `S₂²`) sur une échelle caractéristique **ξ**
(longueur de corrélation, en µm), calculable efficacement par FFT (théorème de Wiener–Khinchin).
`ξ` est **indépendant** de `S` : on peut avoir `S` global faible avec de grands domaines locaux
ordonnés. Dans le générateur, `ξ` est le paramètre de conception de la structure de domaines
(taille des domaines orientés).

### 2.5 Morphologie de fibre : chaîne vermiforme et crimp

Une fibre n'est pas un segment droit. Deux modèles d'ondulation, **physiquement distincts** :

1. **Chaîne vermiforme** (worm-like chain, Kratky–Porod) — ondulation **stochastique** sans échelle
   privilégiée, caractérisée par la **longueur de persistance `Lp`** :

   ```
   ⟨ t(s)·t(s+Δ) ⟩ = exp(−Δ / Lp)
   ```

   où `t(s)` est la tangente à l'abscisse curviligne `s`. `Lp → ∞` : fibre raide/droite ;
   `Lp` petit : fibre très ondulée. Discrétisé : à chaque pas `ds`, la direction subit une rotation
   aléatoire d'écart-type `√(2 ds / Lp)` (c'est exactement le bruit ajouté dans la croissance,
   §3.3).

2. **Crimp** — ondulation **périodique** déterministe, signature du tendon (et de la corde vocale),
   déplacement latéral sinusoïdal d'**amplitude `a_c`** et **période `Λ_c`** :

   ```
   p(s) ← p(s) + a_c · sin(2π s / Λ_c) · e₁     (e₁ ⟂ axe de la fibre)
   ```

Les deux mécanismes se combinent (mettre `a_c = 0` ou `Lp = ∞` désactive l'un ou l'autre). La
**tortuosité** `τ = L/D` (longueur curviligne / distance bout-à-bout) et la **rectitude** `s = D/L`
en découlent et sont mesurées par la Famille F.

### 2.6 Défauts topologiques

Dans un champ directeur nématique, les **disclinaisons** (défauts topologiques de charge ±1/2 en
2D) sont des singularités d'orientation détectables par le **nombre d'enroulement** (winding
number) le long d'un contour fermé. Leur **densité** (nombre par unité d'aire) est un descripteur
puissant des tissus de type cristal-liquide (arcades du cartilage, cornée). Le générateur ne les
*impose* pas explicitement pour l'instant (ils **émergent** des champs et des mélanges de
populations) ; ils sont **mesurés** comme vérité-terrain par la Famille G.

### 2.7 Volume exclu, réticulation, hiérarchie

- **Volume exclu / packing.** Des fibres réelles ne s'interpénètrent pas. La théorie d'Onsager des
  bâtonnets durs relie fraction volumique, anisotropie et transition d'ordre. Conséquence pratique
  pour la génération : avec exclusion, la **fraction volumique plafonne** (limite de packing), et
  l'arrangement latéral devient ordonné (cornée : réseau quasi-cristallin hexagonal).

- **Réseau & réticulation.** Beaucoup de tissus (derme, gels) forment des **réseaux connectés** :
  fibres qui se **ramifient** et **points de réticulation** (crosslinks) qui relient des fibres
  voisines. Ces caractéristiques (points de branchement, de croisement, connectivité) sont
  essentielles à la mécanique et à la texture, et constituent la Famille F « réseau ».

- **Hiérarchie.** Le collagène est **hiérarchique** : molécule → microfibrille → **fibrille** →
  **fibre** → **fascicule** → tendon. Le générateur modélise les trois niveaux supérieurs pertinents
  à l'échelle de l'image (fibrille → fibre → fascicule), reproduisant le *bundling* (regroupement en
  faisceaux) caractéristique du tendon.

---

## 3. Architecture du générateur (principe algorithmique)

Vue d'ensemble : **(1)** un champ d'orientation moyen `n₀(r)` encode la macro-architecture ;
**(2)** chaque fibrille reçoit une direction propre par échantillonnage d'une dispersion biaxiale
autour de `n₀` ; **(3)** la fibrille est **crue** (croissance) en suivant le champ + ondulation ;
**(4)** elle est **rastérisée** en tube binaire plein ; les features réseau (exclusion,
branchement, réticulation, hiérarchie) modulent le placement et la croissance ; **(5)** la
vérité-terrain (tenseur `Q`, `S`, `β`, `ξ`, φ_v) est calculée. Déterministe pour `{config, graine}`.

### 3.1 Champ d'orientation moyen `n₀(r)` — les 6 archétypes

`n₀` est une **fonction de l'espace** ; les fibres la *suivent*, donc les architectures courbes
produisent naturellement des fibres courbes. Implémentation : champs de directeur (`architecture.py`).

| Archétype | `n₀(r)` | Paramètres propres | Tissu cible |
|---|---|---|---|
| **Uniaxial** | direction constante `(φ₀, θ₀)` | φ₀, θ₀ | tendon |
| **Lamellaire** | azimut en marches selon z : `φ(z) = φ_start + Δφ·⌊z/t_lam⌋` | épaisseur `t_lam`, pas `Δφ` (90°) | cornée |
| **Arcade** | élévation graduée : `θ(z)` interpolé de `θ_surface` (z=0) à `θ_deep` (z max) | θ_deep, θ_surface | cartilage (arcades de Benninghoff) |
| **Tubulaire** | circonférentiel autour d'un axe : `n = cosβ·ê_circ + sinβ·ẑ` | angle d'hélice `β`, option **croisé ±β** | paroi d'artère (média/adventice), anneau de disque |
| **Biaxial** | mélange de **2 populations** uniaxiales `(φ_a, φ_b)` | φ_a, φ_b, fraction de mélange | peau (lignes de Langer, vannerie) |
| **Isotrope** | non défini ; `κ ≡ 0` impose l'isotropie | — | tissu très désorganisé |

Les architectures *multi-axes* (biaxial, tubulaire croisé) sont représentées par **plusieurs
populations pondérées**, chacune avec son propre champ ; chaque fibrille est tirée dans une
population selon les poids.

### 3.2 Dispersion biaxiale (échantillonnage Watson/Bingham)

Autour de la direction locale `n₀`, chaque fibre reçoit une **déviation** échantillonnée avec une
loi axiale anisotrope, paramétrée par `(κ∥, κ⊥)` :

1. **angle polaire `γ`** (écart à l'axe) tiré d'une **loi de Watson axiale** gouvernée par la
   concentration la plus faible `κ_polaire = min(κ∥, κ⊥)` (la direction la plus dispersée fixe
   l'étalement) : on échantillonne `u = cos γ ∈ [0,1]` de densité `∝ exp(κ_polaire·u²)` (rejet),
   d'où `γ = arccos u`. `κ = 0` → `u` uniforme → `γ` distribué en `sin γ` (**isotrope** exact) ;
   `κ` grand → `γ` petit (serré) ;
2. **azimut `ψ`** de la déviation dans le plan tangent, **concentré vers l'axe de plus faible κ**
   (von Mises sur `2ψ`, concentration `∝ |κ⊥ − κ∥|`) → étalement anisotrope (biaxial) ;
3. **signe aléatoire** (la fibre est axiale).

Ce schéma reproduit fidèlement les limites : `κ∥ = κ⊥` grand → cluster serré (tendon) ;
`κ∥ ≪ κ⊥` → girdle planaire (peau) ; `κ → 0` → isotrope. La **vérité-terrain** `S, β` étant
**mesurée** sur l'échantillon généré (et non supposée à partir de κ), aucune inversion analytique
fragile n'est nécessaire.

### 3.3 Croissance des centerlines

Une fibrille démarre d'un point-germe `p₀` et croît **bidirectionnellement**. À chaque pas
(longueur `ds`), la direction est :

```
d = normalize( (n₀(p) + offset)·sign + bruit_WLC )
p ← p + d·ds
```

- `offset = base_dir − n₀(p₀)` : inclinaison **constante** propre à la fibre (= sa déviation de
  dispersion), ajoutée au champ local → la fibre suit la courbure du champ tout en gardant son
  inclinaison ; `sign = ±1` pour les deux moitiés (croissance symétrique) ;
- `bruit_WLC ~ N(0, √(2 ds / Lp))` : marche aléatoire persistante (chaîne vermiforme, §2.5) ;
- après croissance, **crimp** appliqué comme déplacement latéral sinusoïdal.

La longueur totale est tirée d'une **loi log-normale** (moyenne `L̄`, CV) ; une fibre de longueur
nulle « traverse le volume » (longueur = diagonale).

### 3.4 Rastérisation en tubes binaires

Chaque fibrille est peinte comme une **capsule** (cylindre à bouts hémisphériques) : un voxel est
marqué `1` si sa **distance physique au segment de centerline ≤ rayon = diamètre/2**. On obtient un
**volume binaire d'occupation** (1 = intérieur de fibrille, 0 = vide) — de *vraies* structures
tubulaires 3D, pas des points. Le champ directeur du voxel = tangente locale ; la polarité = celle
de la fibre. Le diamètre est tiré d'une loi normale (moyenne `d̄`, CV).

### 3.5 Volume exclu

Si activé : un volume **`owner [Z,Y,X]`** mémorise quelle fibrille possède chaque voxel.

- **Germination** : les germes sont rejetés s'ils tombent dans un voxel déjà occupé (placement en
  espace libre) ;
- **Croissance** : une fibre **s'arrête** dès qu'elle entre dans un voxel possédé par une autre
  fibre (pas d'interpénétration) ;
- **Rastérisation** : seuls les voxels **libres** sont revendiqués.

Conséquence : la fraction volumique **plafonne** (limite de packing), des **interstices**
subsistent, et chaque voxel a un propriétaire unique (réseau non chevauchant).

### 3.6 Branchement et réticulation

- **Branchement** : le long d'une fibre (longueur `L`), on tire `n ~ Poisson(densité_branche · L)`
  points de branchement ; chaque enfant croît depuis le point parent, dans une direction = tangente
  parente **déviée de l'angle de branchement** (rotation autour d'un axe ⟂ aléatoire), avec un
  diamètre et une longueur réduits. Récursif, **profondeur limitée** (anti-explosion). Les enfants
  héritent de `fiber_id`/`fascicle_id` (lien hiérarchique) et portent `type="branch"`.
- **Réticulation (crosslinks)** : après placement, un **arbre k-d** (cKDTree) indexe tous les points
  de centerline ; on tire des paires de points **proches mais de fibres différentes** (distance ≤
  `crosslink_max_um`) et on insère un **connecteur court** (`type="crosslink"`) entre eux. Cela crée
  la connectivité de réseau et les points de croisement.

### 3.7 Hiérarchie fascicule → fibre → fibrille

Si activée, le placement plat est remplacé par un placement **imbriqué** :

```
pour chaque fascicule (centre fc, direction = n₀(fc) + dispersion) :
   pour chaque fibre (centre fbc = fc + bille(rayon_fascicule), direction = dir_fascicule + dispersion κ_fibre) :
      pour chaque fibrille (germe = fbc + bille(rayon_fibre), direction = dir_fibre + dispersion serrée) :
         croître + rastériser ; assigner fiber_id, fascicle_id
```

`fiber_kappa` contrôle l'alignement **intra-fibre** (élevé → fibrilles bien parallèles dans une
fibre). Les centres sont **bornés dans le volume** (clip) pour ne pas perdre de faisceaux.
`n = n_fascicules × fibres_par_fascicule × fibrilles_par_fibre` (remplace le compte plat).

### 3.8 Calcul de la vérité-terrain

Sur l'ensemble des directions tirées (fibres + branches), on calcule :

```
Q = ⟨(3 n⊗n − I)/2⟩ ;  λ₁≥λ₂≥λ₃ ⇒  S = λ₁ ,  β = λ₂ − λ₃ ,  (φ₀, θ₀) = vecteur propre de λ₁
S₂ = |⟨e^{i 2φ}⟩| (in-plane) ;  κ (von Mises) via I₁/I₀ ;  ξ (paramètre de conception)
φ_v (fraction volumique atteinte) = moyenne de l'occupation binaire
```

Ces grandeurs sont stockées dans le phantom (`ground_truth.global`) et constituent les **cibles**
que l'onglet Analyse cherchera à retrouver — fermant la boucle.

---

## 4. Catalogue exhaustif des paramètres

### Bloc 1 — Volume imagé (géométrie de la ROI)

| Paramètre | Symbole | Unité | Rôle | Remarque |
|---|---|---|---|---|
| Taille X/Y/Z | — | µm | étendue physique du volume | — |
| Voxel X/Y/Z | (dx,dy,dz) | µm | échantillonnage spatial | < diamètre pour résoudre les fibrilles |
| # voxels | — | — | `round(Taille/Voxel)` | **dérivé** (non éditable) |

### Bloc 2 — Fibrille individuelle (morphologie d'une fibre)

| Paramètre | Symbole | Unité | Rôle | Plage typique |
|---|---|---|---|---|
| Diamètre moyen | d̄ | µm | épaisseur des tubes | 0.01–1 (collagène : 10 nm–1 µm) |
| Diamètre CV | — | — | dispersion relative du diamètre (écart-type/moyenne) | 0–0.5 |
| Longueur moyenne | L̄ | µm | longueur des fibres (0 = traverse le volume) | 1–∞ |
| Longueur CV | — | — | dispersion relative de longueur | 0–0.5 |
| Persistance | Lp | µm | raideur (chaîne vermiforme) ; ∞ = droit | 5–10⁶ |
| Crimp amplitude | a_c | µm | amplitude d'ondulation périodique (tendon) | 0–5 |
| Crimp période | Λ_c | µm | période de l'ondulation | 10–50 |
| Quantité | N **ou** φ_v | — | nombre de fibrilles **ou** fraction volumique (l'un dérive l'autre) | φ_v 0.01–0.6 |

### Bloc 3 — Organisation du réseau (les descripteurs à retrouver)

| Paramètre | Symbole | Unité | Rôle | Plage |
|---|---|---|---|---|
| Architecture | — | — | macro-organisation `n₀(r)` (6 archétypes) | catégoriel |
| Azimut moyen | φ₀ | ° | direction dominante dans le plan | 0–180 |
| Élévation moyenne | θ₀ | ° | inclinaison hors-plan | −90–90 |
| Axes A/B + mélange | φ_a, φ_b | ° | (biaxial) deux directions + fraction | 0–180 |
| Épaisseur lamelle / Δφ | t_lam, Δφ | µm, ° | (lamellaire) période en z et incrément d'azimut | t : 0.1–50 ; Δφ : 0–180 |
| θ profond / surface | — | ° | (arcade) gradient d'élévation en z | −90–90 |
| Angle d'hélice / croisé | β | ° | (tubulaire) inclinaison circonférentielle, ± symétrique | −90–90 |
| Dispersion in-plane | κ∥ | — | concentration d'orientation dans le plan d'alignement | 0–300 |
| Dispersion out-of-plane | κ⊥ | — | concentration hors plan | 0–300 |
| Longueur de corrélation | ξ | µm | échelle spatiale de l'ordre (local vs global) | 0.1–1000 |
| Graine | seed | — | germe du RNG (reproductibilité) | entier |

> **κ∥ = κ⊥ grands** → cluster serré (tendon) · **κ⊥ ≫ κ∥** → girdle planaire (peau) ·
> **κ → 0** → isotrope. **ξ** grand → ordre global (tendon) ; ξ petit → multi-domaine (derme).

### Bloc « Network features »

| Paramètre | Unité | Rôle |
|---|---|---|
| Volume exclusion | (bool) | empêche l'interpénétration ; plafonne le packing (cornée, tendon) |
| Branch density | /µm | taux de branchement le long des fibres (Poisson) |
| Branch angle | ° | déviation angulaire des branches |
| Crosslink density | /µm³ | densité de connecteurs entre fibres proches |
| Crosslink max dist | µm | distance max pour relier deux fibres |
| Hierarchy | (bool) | active le placement fascicule→fibre→fibrille |
| Fascicles | — | nombre de fascicules |
| Fibers / fascicle | — | fibres par fascicule |
| Fibrils / fiber | — | fibrilles par fibre |

---

## 5. Correspondance tissus → paramètres

| Tissu | Architecture | Dispersion | ξ | Morphologie | Features réseau |
|---|---|---|---|---|---|
| **Tendon** | uniaxial | κ∥, κ⊥ élevés | grand | crimp (a_c, Λ_c), Lp élevé | hiérarchie (fascicules), exclusion |
| **Peau / derme** | biaxial (Langer) ou isotrope | κ⊥ élevé, κ∥ faible (girdle) ; ou κ→0 | petit (multi-domaine) | Lp modéré | branchement + réticulation (vannerie) |
| **Cornée** | lamellaire (Δφ≈90°) | élevé dans la lamelle | ~ épaisseur lamelle | diamètre fin, droit | exclusion (packing serré) |
| **Cartilage** | arcade (θ: 90°→0°) | modéré | modéré | Lp | (défauts d'arcade émergents) |
| **Paroi d'artère** | tubulaire (β, croisé ±β) | modéré | modéré | Lp | exclusion |
| **Os (lamellaire/ostéonal)** | lamellaire ou tubulaire (concentrique) | variable | — | — | exclusion |

Cette table illustre la **complétude** : un seul générateur paramétrique couvre l'éventail des
architectures connues, par simple changement de l'archétype et des dispersions/échelles.

---

## 6. Comparaison avec l'état de l'art

### 6.1 Les trois familles d'approches

1. **Générateurs appris (VAE/GAN/diffusion)** pour images SHG/histologie. Le plus proche : un VAE
   génère des centerlines à topologie contrôlée (orientation, alignement, densité, ondulation,
   longueur) puis un cGAN rend des images réalistes (Liu *et al.*, *Medical Image Analysis*, 2023).
   Édition de features SHG par StyleGAN2-ADA + SeFa, validée par KID (*J. Phys. Photonics*, 2026) ;
   rendu histologique photoréaliste sans IA générative (SYNTA, 2022).
2. **Générateurs procéduraux/stochastiques** (biomécanique) : réseaux de fibres discrets par
   **tessellation de Voronoï/Delaunay** autour de germes aléatoires, avec **distribution von Mises
   + ondulation/crimp + densité + réticulations** (modèles d'adventice, de veines, de gels ;
   *Biomech. Model. Mechanobiol.* 2019 ; *PLOS One* 2014 ; *BioMed. Eng. OnLine* 2011).
3. **Modèles physiques directs SHG** (Lin & Campagnola ; μMAPPS phasor) : estiment diamètre,
   fraction de remplissage, orientation depuis le signal — modélisent le *signal*, pas la *structure*.

### 6.2 Tableau comparatif

| Capacité | Appris (GAN/VAE) | Procédural biomeca | **FibreMap (nous)** |
|---|---|---|---|
| Orientation moyenne + dispersion | ✔ (appris) | ✔ (von Mises) | ✔ **+ biaxial Bingham (κ∥/κ⊥)** |
| Ondulation (Lp, crimp) | ✔ | ✔ | ✔ |
| Diamètre/longueur, densité | ✔ | ✔ | ✔ |
| **Multi-architectures nommées** (lamelles, arcades, tubulaire, Langer) | partiel/2D | ✗ (mono-distribution) | ✔ **unifié 3D** |
| Longueur de corrélation ξ explicite | rare | rare | ✔ |
| **Vérité-terrain d'organisation connue + boucle fermée** | ✗ (FID/KID/expert) | partiel | ✔ **(tenseur Q, S, β, ξ)** |
| Volume exclu / packing | ✗ | ✔ | ✔ |
| Branchement / réticulation / connectivité | ✗ | ✔ | ✔ |
| Hiérarchie (fibrille→fibre→fascicule) | ✗ | partiel | ✔ |
| **Photoréalisme d'image** | ✔✔ | ✗ | ⚠ (Tier 1 scalaire ; Tier 2 à entraîner) |
| Ordre latéral quasi-cristallin (cornée) | ✗ | partiel | ✗ (à venir) |
| Équilibrage mécanique (alignement sous tension) | ✗ | ✔ (certains) | ✗ (hors périmètre) |

### 6.3 Verdict

- **Aussi bien** que les modèles structuraux biomécaniques sur la **morphologie fibrillaire** et les
  **statistiques d'orientation** (et désormais sur packing/branchement/hiérarchie).
- **Mieux** sur deux axes structurels : (i) un **générateur paramétrique unifié** couvrant
  tendon/peau/cornée/cartilage/artère via l'abstraction « champ de directeur » (la littérature est
  surtout mono-tissu ou réseaux isotropes de Voronoï) ; (ii) la **boucle fermée quantitative** avec
  vérité-terrain d'organisation **connue** (tenseur `Q`) — la communauté valide surtout
  qualitativement ou par FID/KID. Comparer des métriques sur **vérité connue** est précisément le
  manque identifié dans la littérature.
- **Moins bien** sur le **photoréalisme d'image** (les GAN/diffusion sont devant — c'est l'objet du
  Tier 2 à entraîner sur données réelles), et sur quelques patterns fins encore absents :
  **ordre latéral hexagonal** (cornée quasi-cristalline), **entrelacement 3D vrai** (vannerie du
  derme au-delà de deux populations), **organisation pilotée mécaniquement**.

---

## 7. Limites actuelles et perspectives

1. **Défauts topologiques** seulement *émergents* (mesurés, pas imposés) → ajouter une seeding
   explicite de disclinaisons ±1/2 (cornée, arcades).
2. **Ordre latéral / packing hexagonal** non modélisé (l'exclusion empêche le chevauchement mais ne
   force pas l'ordre quasi-cristallin de la cornée) → packing dirigé.
3. **Entrelacement 3D** du derme limité à un mélange de 2 populations → modèle de tissage explicite.
4. **Profil-z du tenseur `Q`** : pour quantifier lamelles/arcades, l'analyse devra mesurer `Q(z)`
   (extension prévue côté Livrable 3).
5. **Performance** : la rastérisation des capsules est en boucle CPU (lente sur gros volumes) →
   vectorisation / Numba / GPU.
6. **Photoréalisme** (Tier 2) : diffusion conditionnelle entraînée sur images réelles, en préservant
   la vérité-terrain d'organisation (prochaine étape, onglet 2).

---

## 8. Annexe — formules clés et glossaire

**Directeur** `n = (cosθcosφ, cosθsinφ, sinθ)`, axial (`n ≡ −n`), φ∈[0,π), θ∈[−π/2,π/2].
**Angle doublé** `(cos 2φ, sin 2φ)` pour statistiques circulaires axiales.
**Tenseur d'ordre** `Q = ⟨(3 n⊗n − I)/2⟩` ; `S = λ₁`, `β = λ₂ − λ₃`, direction = vec. propre de λ₁.
**Ordre 2D** `S₂ = |⟨e^{i2φ}⟩|`. **Watson** `∝ exp(κ(μ·n)²)` ; **Bingham** `∝ exp(nᵀBn)`.
**Corrélation** `C(r) = ⟨P₂(n(x)·n(x+r))⟩` → plateau `S²`, échelle `ξ`.
**WLC** `⟨t(s)·t(s+Δ)⟩ = exp(−Δ/Lp)` ; pas discret : rotation d'écart-type `√(2ds/Lp)`.
**Crimp** déplacement `a_c sin(2πs/Λ_c)`. **Tortuosité** `τ = L/D`.

| Terme | Définition |
|---|---|
| **ODF** | fonction de distribution d'orientation (sur la sphère/le cercle) |
| **Nématique** | phase ordonnée en orientation (axiale) sans ordre positionnel |
| **Girdle** | distribution en ceinture (confinée à un plan, dispersée dans ce plan) |
| **Disclinaison** | défaut topologique du champ directeur (charge ±1/2 en 2D) |
| **Fascicule** | faisceau de fibres ; niveau hiérarchique supérieur du tendon |
| **Vérité-terrain** | valeurs d'organisation **connues** par construction (cibles de l'analyse) |

### Références indicatives
- Liu *et al.*, « Collagen Fiber Centerline Tracking… VAE-based Synthetic Training Data »,
  *Medical Image Analysis*, 2023.
- « Artificial SHG image feature tuning with generative models and SeFa », *J. Phys. Photonics*, 2026.
- SYNTA, « deep learning-based image analysis… photo-realistic synthetic data », 2022.
- « Micro-mechanics of collagen fiber network in the tunica adventitia », *Biomech. Model.
  Mechanobiol.*, 2019.
- « A three-dimensional computational model of collagen network mechanics » (Voronoï), *PLOS One*, 2014.
- « Structural constitutive model considering angular dispersion and waviness of collagen fibres »,
  *BioMed. Eng. OnLine*, 2011.
- « Imaging and modeling collagen architecture from the nano to micro scale », 2014.
- de Gennes & Prost, *The Physics of Liquid Crystals* (tenseur d'ordre nématique).
- Mardia & Jupp, *Directional Statistics* (Watson, Bingham, von Mises–Fisher).