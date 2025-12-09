🚀 CFD–ML Aero Project Portfolio

Author: Aditya
Tech Stack: ANSYS Fluent, CFD, Turbulence Modeling, Python, NumPy, TensorFlow, PyVista, MeshIO, VTK, Git, JupyterLab
Status: Active Development (2025–2026)

📌 1. Project Overview

This repository documents a full end-to-end CFD + Machine Learning workflow applied to aerospace and propulsion problems.
The goal is to build a strong, industry-relevant portfolio targeting roles in:

🚀 Aerospace Startups (Skyroot, Agnikul, Bellatrix)

✈️ Aerodynamics / Propulsion CFD Roles

🔥 High-Fidelity CFD Research

🤖 Physics-ML / Surrogate Modeling

This repo contains:

High-quality ANSYS Fluent cases

Clean Python post-processing tools

Reproducible ML models for CFD datasets

Professional project structure & documentation

📁 2. Repository Structure
cfd-ml-project/
│
├── ansys_cases/         # All Fluent cases (mesh, setup, results)
│   ├── setup/
│   └── results/
│
├── data/
│   ├── raw/             # Raw simulation exports
│   ├── processed/       # Cleaned & structured data for ML
│   └── meshes/          # Mesh files (.msh/.cas.h5)
│
├── notebooks/           # JupyterLab notebooks for analysis/ML
├── scripts/             # Python automation & utilities
├── utils/               # Helper modules (mesh readers, VTK utils)
│
├── models/              # ML models
│   ├── checkpoints/
│   └── final/
│
├── tests/               # Environment & code sanity tests
│
├── environment.yml      # Fully reproducible Python environment
├── run_test.py          # Quick health check for setup
└── README.md            # This file

🧪 3. Environment Setup

Recreate this environment on any machine:

conda env create -f environment.yml
conda activate cfdml


To validate the installation:

python run_test.py


Expected output:
Environment smoke test: OK ✓

🧰 4. Tools & Technologies
CFD Tools

ANSYS Fluent 2023R1 (validated)

Turbulence Models: k-ω SST, k-ε, RANS

Geometry + meshing (structured/unstructured)

Python / ML Stack

NumPy, Pandas

TensorFlow 2.14

Matplotlib

PyVista (3D visualization)

MeshIO, VTK, scikit-image

Dev Tools

Git + GitHub

JupyterLab

Windows + Conda environment

📘 5. Planned Projects (Recruiter-Friendly & Aero-Relevant)
✅ 1) Airfoil CFD Dataset + ML Surrogate (Regression)

2D airfoil mesh in Fluent

Pressure, velocity, lift/drag for various AoA

Train ML model to predict Cp distribution

(High value for aerodynamics roles)

✅ 2) Rocket Nozzle Internal Flow + Thrust Prediction

Axisymmetric nozzle CFD

Vary chamber pressure & expansion ratio

Train model to estimate thrust & exit Mach

(Directly relevant to Skyroot/Agnikul)

✅ 3) Heat Transfer in a Cooling Channel

Conjugate heat transfer (CHT)

Predict wall temperature distribution

(Good for thermal + ML hybrid profiles)

✅ 4) Mesh-to-Field Super-Resolution Model

Up-sampling coarse CFD results → fine grid

Uses CNN / U-Net architecture

(Trending Physics-ML topic)

🏁 6. Current Progress
Component	Status
GitHub project setup	✔ Complete
Conda environment + testing	✔ Complete
Fluent installation verified	✔ Complete
CFD cases	🔄 In progress
ML notebooks	⏳ Scheduled
Final models	⏳ Upcoming
🎯 7. Target Roles

This portfolio is built for roles like:

CFD Engineer

Aerospace Simulation Engineer

Aerodynamics Engineer

Propulsion CFD Engineer

ML for Physics Engineer

🔗 8. Contact

For collaboration or opportunities:

Aditya
📧 adityakarri39@gmail.com
📍 hyderabad | Open to relocation