# Cartesian SO(3)-Equivariant Vector-Scalar MLIP on QM9

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/milestone1_qm9_equivariant_mlip.ipynb)

**Course:** Deep Learning (`BMEVIMIAD04`) — Budapest University of Technology and Economics (BME VIK)  
**Project Title (EN):** *Cartesian SO(3)-Equivariant Vector-Scalar Message Passing Neural Network for Quantum-Chemical Potential Energy Surfaces*  
**Project Title (HU):** *Descartes-féle SO(3)-ekvivariáns vektor-skalár üzenettovábbító neurális hálózat kvantumkémiai potenciálisenergia-felületek közelítésére*

---

## 1. Team Information & Role Rotation Matrix

| Team Name | **EquiChem-DL** |
| :--- | :--- |
| **Member 1** | `[Name 1]` — Neptun Code: `[XXXXXX]` |
| **Member 2** | `[Name 2]` — Neptun Code: `[YYYYYY]` |
| **Member 3** | `[Name 3]` — Neptun Code: `[ZZZZZZ]` |

In accordance with the course rules (each member covers at least 3 areas, and every member participates in task 3 or 4), the project responsibilities are distributed as follows:

| Task Area | Member 1 (`XXXXXX`) | Member 2 (`YYYYYY`) | Member 3 (`ZZZZZZ`) |
| :--- | :---: | :---: | :---: |
| **1. Literature Review & Quantum-Chemical Theory** | ✅ | ✅ | — |
| **2. Data Acquisition, Cleaning & Equivariant Graph Preprocessing** | ✅ | — | ✅ |
| **3. Neural Network Architecture & Training Pipeline** | ✅ | ✅ | — |
| **4. Hyperparameter Optimization, Evaluation & Testing** | — | ✅ | ✅ |
| **5. Interactive 3D Visualization / Interface Integration** | — | — | ✅ |

---

## 2. Project Overview

Approximating *ab initio* Density Functional Theory (DFT) Potential Energy Surfaces (PES) using Machine Learning Interatomic Potentials (MLIPs) accelerates quantum-chemical property prediction by five orders of magnitude. While first-generation invariant Graph Neural Networks (such as SchNet) rely solely on pairwise interatomic distances $d_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|_2$ and are blind to three-body bond angles $\theta_{jik}$ within a single hop, explicit triplet networks (such as DimeNet) suffer from $\mathcal{O}(N\bar{k}^2)$ complexity.

This project implements a from-scratch **Cartesian $SO(3)$-Equivariant Vector-Scalar Message Passing Neural Network** on the **QM9 benchmark dataset** ($130,831$ characterized equilibrium organic molecules containing $\text{H, C, N, O, F}$ with up to $29$ atoms).
* **Coupled Vector-Scalar Hidden States:** Each atom maintains both rotation-invariant scalar channels ($\mathbf{s}_i \in \mathbb{R}^{F}$) and rotation-equivariant 3D Cartesian vector channels ($\vec{\mathbf{v}}_i \in \mathbb{R}^{F \times 3}$). Contracting projected vector channels via Euclidean inner products ($\langle \mathbf{U}_1 \vec{\mathbf{v}}_i, \mathbf{U}_2 \vec{\mathbf{v}}_i \rangle_{\mathbb{R}^3}$) automatically recovers $\cos(\theta_{jik})$ bond-angle geometry in strictly linear $\mathcal{O}(N\bar{k})$ edge complexity.
* **Leak-Free Empirical OLS Baseline (`Atomref`) Removal:** Instead of relying on external offset tables, we fit an Ordinary Least Squares (OLS) linear model on the training set stoichiometry matrix (`torch.linalg.lstsq`), reducing the target $U_0$ energy standard deviation from $1,087.3\text{ eV}$ to $2.41\text{ eV}$ ($450\times$ reduction).
* **Sparse Size-Independent Disjoint-Graph Batching:** Zero-padding is completely avoided. Variable-sized molecules are collated into a block-diagonal sparse super-graph controlled by a dynamic atom-budget sampler (`MaxAtomsBatchSampler`), enabling seamless inference and training on macromolecules with $N > 29$ atoms.

---

## 3. Repository File Structure

| File Name | Description |
| :--- | :--- |
| `milestone1_qm9_equivariant_mlip.ipynb` | **Main Milestone 1 Jupyter Notebook.** Contains the scientific introduction, automated QM9 download and cleaning script, 10-panel Exploratory Data Analysis (EDA) and 3D molecular visualizations, leak-free 80/10/10 splitting with OLS baseline fitting, custom $SO(3)$-equivariant directional graph builder, sparse disjoint-graph `DataLoader` pipelines, and $SO(3)$ symmetry + macromolecular ($N=150$) verification tests. |
| `beszamolo.tex` / `beszamolo.pdf` | **Bilingual Academic Project Report (Beszámoló dokumentum).** Formatted in two-column IEEE/NeurIPS academic style, containing the complete English report followed by the complete Hungarian translation and 14 peer-reviewed academic references. |
| `README.md` | **Repository Documentation.** Team details, architectural summary, file index, and reproduction instructions. |

---

## 4. Software Environment & Execution Instructions

### Option A: 1-Click Execution in Google Colab (Recommended)
1. Click the **`Open in Colab`** badge at the top of this `README.md` or inside `milestone1_qm9_equivariant_mlip.ipynb` (replace `YOUR_GITHUB_USERNAME/YOUR_REPO_NAME` in the badge URL with your repository path once pushed).
2. In Google Colab, select **Runtime $\rightarrow$ Run all**.
3. The notebook automatically installs `torch_geometric` if not present, downloads and caches the QM9 dataset in ~10 seconds, generates all 3 multi-panel publication figures, saves the deterministic split indices (`./processed_splits/qm9_splits_and_atomrefs.npz`), and verifies the prepared Training, Validation, and Test batches.

### Option B: Local Execution
Ensure Python $\ge 3.10$ is installed, then install the dependencies and launch Jupyter:

```bash
pip install torch torchvision torchaudio torch_geometric matplotlib numpy
jupyter notebook milestone1_qm9_equivariant_mlip.ipynb
```

### Reproducibility Parameters
* **Random Seed:** `SEED = 42` (enforced across `random`, `numpy`, and `torch`).
* **Dataset Partition:** `80%` Training ($104,664$ molecules), `10%` Validation ($13,083$ molecules), `10%` Test ($13,084$ molecules), excluding the $3,054$ uncharacterized GDB-9 geometries.
* **Spatial Cutoff Radius:** $r_c = 5.0\text{ \AA}$ with $K = 20$ smooth Gaussian Radial Basis Functions and a cosine cutoff envelope.
* **Dynamic Batching Budget:** `max_atoms = 1800`, `max_molecules = 100`.