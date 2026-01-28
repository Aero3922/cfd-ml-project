🚀 CFD–ML Aero Project Portfolio

**Author:** Aditya subhash reddy karri

**Location:** Hyderabad (open to relocation)
**Contact:** [adityakarri39@gmail.com](mailto:adityakarri39@gmail.com)

**Tech Stack:**
ANSYS Fluent · CFD · Turbulence Modeling · Python · NumPy · TensorFlow · PyVista · MeshIO · VTK · Git · JupyterLab

**Status:** Active development — **CFD workflows implemented, ML extensions ongoing**

---

## 📌 1. Overview
This repository documents a **structured, end-to-end CFD workflow with machine-learning extensions**, focused on **aerospace and propulsion applications**.
The primary emphasis of this work is **CFD correctness and physical understanding** (geometry, meshing, turbulence modeling, solver setup, and post-processing).
Machine learning components are being **incrementally developed on top of validated CFD data** to explore surrogate modeling and acceleration use cases.

This portfolio is intended to demonstrate:
* Practical CFD capability using **ANSYS Fluent**
* Clean, reproducible **Python-based post-processing**
* A disciplined pathway from **physics-based simulation → data → ML**

## 🎯 2. Target Roles

This portfolio is aligned with entry-level to early-career roles such as:
* CFD Engineer
* Aerodynamics Engineer
* Aerospace Simulation Engineer
* Propulsion / Internal Flow CFD Engineer
* Physics-ML / Simulation Acceleration Engineer

## 📁 3. Repository Structure

```
cfd-ml-project/
│
├── ansys_cases/          # Fluent cases (geometry, mesh, setup)
│   ├── setup/
│   └── results/
│
├── data/
│   ├── raw/              # Raw CFD exports
│   ├── processed/        # Cleaned datasets for analysis / ML
│   └── meshes/           # Mesh files (.msh / .cas.h5)
│
├── notebooks/            # Jupyter notebooks (CFD analysis, ML experiments)
├── scripts/              # Python automation & utilities
├── utils/                # Helper modules (VTK, mesh readers, etc.)
│
├── models/
│   ├── checkpoints/
│   └── final/
│
├── tests/                # Environment & sanity checks
│
├── environment.yml       # Reproducible Python environment
├── run_test.py           # Environment smoke test
└── README.md
```

## 🧪 4. Environment Setup
The Python environment is fully reproducible using Conda.
```bash
conda env create -f environment.yml
conda activate cfdml
```

Verify the setup:
```bash
python run_test.py
```
Expected output:

```
Environment smoke test: OK ✓
```
---

## 🧰 5. Tools & Technologies

### CFD & Simulation
* **ANSYS Fluent 2025 R1** (used for all CFD cases)
* External & internal flow simulations
* Turbulence models: **RANS (k-ω SST, k-ε)**
* Structured and unstructured meshing
* Force, pressure coefficient, and field post-processing

### Python / Data / ML
* NumPy, Pandas
* TensorFlow (for surrogate experiments)
* Matplotlib
* PyVista, VTK, MeshIO (CFD data handling & visualization)

### Development
* Git & GitHub
* JupyterLab
* Conda (environment management)
* Windows OS

---

## 📘 6. Implemented & Roadmap Projects

### ✅ 1) Airfoil CFD Dataset & ML Surrogate *(Implemented / In Progress)*
* 2D airfoil simulations in **ANSYS Fluent**
* Lift, drag, and pressure coefficient (Cp) extraction
* Python-based post-processing and automation
* Initial ML experiments to approximate Cp distributions from CFD data

**Status:**
CFD workflow implemented and reproducible
ML surrogate development in progress

### 🔄 2) Rocket Nozzle Internal Flow & Thrust Prediction *(Planned)*
* Axisymmetric internal flow CFD
* Chamber pressure and expansion ratio studies
* Dataset preparation for thrust and Mach number prediction

### ⏳ 3) Heat Transfer in Cooling Channels *(Planned)*
* Conjugate heat transfer (CHT) simulations
* Wall temperature and heat flux analysis
* Potential ML-assisted thermal prediction

### ⏳ 4) Mesh-to-Field Super-Resolution *(Planned Research Extension)*
* Learning fine-grid flow features from coarse CFD results
* Exploration of CNN / U-Net style architectures
* Physics-aware data-driven enhancement

---

## 🏁 7. Current Progress Summary

| Component                   | Status         |
| --------------------------- | -------------- |
| Repository structure        | ✅ Complete     |
| Conda environment & tests   | ✅ Complete     |
| Fluent installation & setup | ✅ Verified     |
| Core CFD cases              | 🔄 In progress |
| ML notebooks                | ⏳ Ongoing      |
| Final surrogate models      | ⏳ Upcoming     |

---

## 📌 8. Notes for Reviewers
This repository is **under active development**.
* CFD workflows are the primary focus and are designed to be **physically sound and reproducible**
* Machine learning components are added **progressively**, with emphasis on interpretability and engineering relevance
* The goal is not to replace CFD, but to **augment it responsibly**
Feedback and discussion are welcome.

## 🔗 9. Contact
For collaboration, discussion, or opportunities:
**Aditya**
📧 [adityakarri39@gmail.com](mailto:adityakarri39@gmail.com)
📍 Hyderabad | Open to relocation