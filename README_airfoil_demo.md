# Airfoil ML Surrogate Inference Demo

This script demonstrates the end-to-end deployment of our trained Machine Learning surrogate models. It bypasses the heavy workload of the ANSYS Fluent CFD solver, providing near-instantaneous aerodynamic pressure distributions and global forces for a given Angle of Attack.

## How It Works
The script loads two distinct production models:
1. **Direct Dense MLP:** Predicts macroscopic forces ($C_l$, $C_d$) with sub-2% error.
2. **Physics-Informed 1D-CNN (PINN):** Predicts the 200-point surface pressure distribution ($C_p$). The PINN was trained with a custom integration loss function to guarantee mathematical momentum conservation across the spatial field.

## Execution Instructions
Ensure your `cfdml` conda environment is active. Run the script via the command line, passing your desired Angle of Attack (AoA) using the `--aoa` flag.

```bash
python predict_airfoil_demo.py --aoa 7.5
(input any aoa between the training envelope -4 and 18)