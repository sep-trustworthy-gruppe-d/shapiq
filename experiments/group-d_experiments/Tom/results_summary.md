# Probly × shapiq: Vision-Ergebnisse

Konfiguration: vortrainierte ImageNet-Modelle, MC-Dropout T=10, p=0.3, 3×3-Grid (N=9), exaktes k-SII bis Ordnung 2 via `ExactComputer` (2^9=512 Koalitionen), Budgetobergrenze=512, Seed=42, CPU-Threads=2.

Credal-Intervalle beziehen sich auf die vorhergesagte Top-1-Klasse; Entropien sind in Nats. Die φ-Matrizen sind zeilenweise nach den neun Grid-Feldern geordnet.

## GoldenRetriever.jpeg — ViT

- Top-1: **golden retriever** (ImageNet-Index 207)
- Mittlere Konfidenz p̄: `0.975057`
- Credal-Intervall: `[0.960442, 0.985139]`; Δp: `0.024697`
- Entropie: H_total `0.175860`; H_alea `0.172941`; H_epi `0.002919`

### 1st-Order φ für `target_probability`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| +0.011977 | +0.512098 | +0.088060 |
| +0.033235 | +0.134280 | +0.035680 |
| +0.008663 | +0.198917 | +0.068754 |

### 1st-Order φ für `epistemic`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| -0.007166 | +0.137446 | +0.076175 |
| +0.109424 | +0.195666 | +0.076752 |
| +0.054317 | +0.079404 | +0.026162 |

### Top-5 Paar-Interaktionen aus k-SII (`target_probability`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,1) ↔ (2,1) | -0.127579 |
| (0,0) ↔ (0,1) | +0.110291 |
| (2,1) ↔ (2,2) | +0.102996 |
| (0,1) ↔ (1,1) | -0.087572 |
| (0,1) ↔ (2,2) | -0.082573 |

### Top-5 Paar-Interaktionen aus k-SII (`epistemic`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,1) ↔ (1,1) | -0.119937 |
| (0,1) ↔ (0,2) | -0.074566 |
| (1,1) ↔ (2,1) | -0.069680 |
| (0,1) ↔ (2,1) | -0.059754 |
| (0,0) ↔ (0,1) | -0.047121 |

### Effizienzprüfung

| Notion | Σ φ_S (S≠∅) | v(voll) − v(∅) | Absolutfehler |
|---|---:|---:|---:|
| target | +0.974531876 | +0.974531876 | 3.331e-16 |
| epistemic | +0.002918795 | +0.002918795 | 7.373e-17 |

Gemessene Laufzeit für Vorhersage und beide Erklärungen: **372.41 s**.

## GoldenRetriever.jpeg — VGG-16

- Top-1: **golden retriever** (ImageNet-Index 207)
- Mittlere Konfidenz p̄: `0.930215`
- Credal-Intervall: `[0.825970, 0.990383]`; Δp: `0.164413`
- Entropie: H_total `0.410820`; H_alea `0.364774`; H_epi `0.046046`

### 1st-Order φ für `target_probability`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| +0.039458 | +0.018357 | +0.069257 |
| +0.017511 | +0.028862 | +0.012967 |
| +0.001320 | +0.015078 | +0.030341 |

### 1st-Order φ für `epistemic`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| +0.041098 | +0.186368 | +0.053610 |
| +0.050955 | +0.102540 | +0.083794 |
| +0.067438 | +0.096386 | +0.112099 |

### Top-5 Paar-Interaktionen aus k-SII (`target_probability`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,1) ↔ (1,1) | +0.459901 |
| (0,1) ↔ (0,2) | +0.212331 |
| (1,0) ↔ (1,1) | +0.098220 |
| (0,2) ↔ (1,1) | -0.084894 |
| (0,0) ↔ (0,1) | +0.082881 |

### Top-5 Paar-Interaktionen aus k-SII (`epistemic`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,1) ↔ (1,1) | -0.201508 |
| (0,1) ↔ (2,1) | -0.074505 |
| (0,1) ↔ (1,2) | -0.043665 |
| (0,1) ↔ (0,2) | -0.042975 |
| (1,1) ↔ (2,0) | -0.041820 |

### Effizienzprüfung

| Notion | Σ φ_S (S≠∅) | v(voll) − v(∅) | Absolutfehler |
|---|---:|---:|---:|
| target | +0.929686890 | +0.929686890 | 0.000e+00 |
| epistemic | +0.012444049 | +0.012444049 | 3.469e-17 |

Gemessene Laufzeit für Vorhersage und beide Erklärungen: **1178.03 s**.

## DogCat.jpeg — ViT

- Top-1: **golden retriever** (ImageNet-Index 207)
- Mittlere Konfidenz p̄: `0.326678`
- Credal-Intervall: `[0.056089, 0.574330]`; Δp: `0.518241`
- Entropie: H_total `3.596630`; H_alea `3.395589`; H_epi `0.201041`

### 1st-Order φ für `target_probability`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| +0.020128 | +0.039502 | +0.122022 |
| +0.018609 | +0.025339 | +0.074112 |
| +0.055940 | +0.050743 | +0.104781 |

### 1st-Order φ für `epistemic`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| +0.015406 | +0.108142 | -0.019358 |
| +0.084815 | +0.158683 | +0.030783 |
| +0.027929 | +0.085137 | -0.001084 |

### Top-5 Paar-Interaktionen aus k-SII (`target_probability`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,1) ↔ (1,1) | +0.272147 |
| (0,1) ↔ (0,2) | -0.178462 |
| (0,2) ↔ (1,1) | -0.171563 |
| (1,0) ↔ (1,1) | +0.099547 |
| (1,1) ↔ (2,1) | +0.087408 |

### Top-5 Paar-Interaktionen aus k-SII (`epistemic`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,1) ↔ (2,1) | -0.088637 |
| (0,0) ↔ (1,1) | -0.057430 |
| (0,1) ↔ (1,1) | -0.055738 |
| (0,1) ↔ (0,2) | +0.052112 |
| (0,1) ↔ (1,0) | -0.046447 |

### Effizienzprüfung

| Notion | Σ φ_S (S≠∅) | v(voll) − v(∅) | Absolutfehler |
|---|---:|---:|---:|
| target | +0.326153087 | +0.326153087 | 1.110e-16 |
| epistemic | +0.201040983 | +0.201040983 | 0.000e+00 |

Gemessene Laufzeit für Vorhersage und beide Erklärungen: **369.92 s**.

## DogCat.jpeg — VGG-16

- Top-1: **golden retriever** (ImageNet-Index 207)
- Mittlere Konfidenz p̄: `0.522001`
- Credal-Intervall: `[0.232071, 0.930888]`; Δp: `0.698817`
- Entropie: H_total `2.169098`; H_alea `1.749127`; H_epi `0.419971`

### 1st-Order φ für `target_probability`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| -0.058710 | -0.094109 | +0.103788 |
| -0.064632 | -0.095011 | +0.028105 |
| +0.001165 | -0.101667 | -0.020861 |

### 1st-Order φ für `epistemic`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| +0.007655 | +0.116596 | -0.040820 |
| +0.077242 | +0.103288 | -0.003330 |
| +0.032111 | +0.099793 | +0.047768 |

### Top-5 Paar-Interaktionen aus k-SII (`target_probability`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,1) ↔ (1,1) | +0.257259 |
| (0,0) ↔ (0,1) | +0.155977 |
| (0,0) ↔ (1,1) | +0.154429 |
| (1,0) ↔ (1,1) | +0.111880 |
| (0,1) ↔ (1,0) | +0.109627 |

### Top-5 Paar-Interaktionen aus k-SII (`epistemic`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,1) ↔ (2,1) | -0.089013 |
| (0,1) ↔ (0,2) | +0.085604 |
| (1,0) ↔ (1,1) | -0.064299 |
| (0,0) ↔ (0,2) | +0.064158 |
| (1,0) ↔ (2,1) | -0.056470 |

### Effizienzprüfung

| Notion | Σ φ_S (S≠∅) | v(voll) − v(∅) | Absolutfehler |
|---|---:|---:|---:|
| target | +0.521472857 | +0.521472857 | 2.220e-16 |
| epistemic | +0.386368871 | +0.386368871 | 5.551e-17 |

Gemessene Laufzeit für Vorhersage und beide Erklärungen: **1958.85 s**.

## blurryFrog.jpeg — ViT

- Top-1: **nematode** (ImageNet-Index 111)
- Mittlere Konfidenz p̄: `0.125048`
- Credal-Intervall: `[0.046178, 0.248361]`; Δp: `0.202182`
- Entropie: H_total `4.674076`; H_alea `4.561948`; H_epi `0.112128`

### 1st-Order φ für `target_probability`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| -0.011389 | +0.009672 | -0.016943 |
| -0.003363 | -0.023669 | -0.008700 |
| -0.002125 | -0.035850 | +0.001048 |

### 1st-Order φ für `epistemic`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| +0.058563 | +0.061125 | +0.008134 |
| +0.043299 | +0.037142 | +0.031835 |
| +0.004306 | -0.004135 | +0.036339 |

### Top-5 Paar-Interaktionen aus k-SII (`target_probability`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,0) ↔ (1,0) | +0.053878 |
| (1,1) ↔ (1,2) | +0.048613 |
| (0,2) ↔ (1,2) | +0.035505 |
| (0,1) ↔ (1,1) | +0.029377 |
| (0,1) ↔ (1,0) | -0.026310 |

### Top-5 Paar-Interaktionen aus k-SII (`epistemic`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,0) ↔ (1,0) | +0.033847 |
| (0,0) ↔ (1,1) | -0.027659 |
| (0,0) ↔ (0,1) | -0.025786 |
| (0,1) ↔ (1,1) | -0.025686 |
| (0,1) ↔ (2,2) | -0.023916 |

### Effizienzprüfung

| Notion | Σ φ_S (S≠∅) | v(voll) − v(∅) | Absolutfehler |
|---|---:|---:|---:|
| target | +0.107871365 | +0.107871365 | 1.388e-17 |
| epistemic | +0.112127781 | +0.112127781 | 0.000e+00 |

Gemessene Laufzeit für Vorhersage und beide Erklärungen: **359.41 s**.

## blurryFrog.jpeg — VGG-16

- Top-1: **jellyfish** (ImageNet-Index 107)
- Mittlere Konfidenz p̄: `0.234286`
- Credal-Intervall: `[0.081585, 0.404724]`; Δp: `0.323139`
- Entropie: H_total `3.138467`; H_alea `2.836297`; H_epi `0.302171`

### 1st-Order φ für `target_probability`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| -0.072604 | -0.071289 | -0.077647 |
| -0.080336 | -0.081219 | -0.071067 |
| -0.060811 | -0.061261 | -0.034930 |

### 1st-Order φ für `epistemic`

| φ(row,col) | 0 | 1 | 2 |
|---:|---:|---:|---:|
| +0.032428 | +0.004970 | +0.028933 |
| +0.037517 | +0.017892 | +0.053878 |
| +0.056454 | +0.064965 | +0.032101 |

### Top-5 Paar-Interaktionen aus k-SII (`target_probability`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (1,0) ↔ (1,1) | +0.035186 |
| (0,0) ↔ (1,0) | +0.034221 |
| (0,2) ↔ (1,0) | +0.034119 |
| (0,0) ↔ (0,2) | +0.034006 |
| (0,1) ↔ (0,2) | +0.033794 |

### Top-5 Paar-Interaktionen aus k-SII (`epistemic`)

| Patchpaar (row,col) | φᵢⱼ |
|---|---:|
| (0,2) ↔ (2,1) | -0.030065 |
| (0,1) ↔ (0,2) | +0.023184 |
| (0,0) ↔ (2,1) | -0.022639 |
| (0,0) ↔ (0,1) | +0.022017 |
| (1,2) ↔ (2,0) | -0.021883 |

### Effizienzprüfung

| Notion | Σ φ_S (S≠∅) | v(voll) − v(∅) | Absolutfehler |
|---|---:|---:|---:|
| target | +0.234210295 | +0.234210295 | 2.776e-17 |
| epistemic | +0.268568277 | +0.268568277 | 0.000e+00 |

Gemessene Laufzeit für Vorhersage und beide Erklärungen: **1428.38 s**.
