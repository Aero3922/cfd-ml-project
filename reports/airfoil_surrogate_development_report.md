# Progress Report: Physics-Informed Neural Network (PINN) for Airfoil Aerodynamics

## 1. Data Generation Approach
CFD data was generated through a parameterized Angle of Attack (AoA) sweep in ANSYS Fluent. The simulations were run using a Steady RANS solver with the SST $k-\omega$ turbulence model at a Reynolds number of 6e6. For each flight condition, macroscopic forces ($C_l$, $C_d$) and surface pressure coefficient ($C_p$) distributions were exported. 

To accurately resolve high aerodynamic gradients, the structured mesh heavily clustered nodes near the airfoil's leading and trailing edges. For machine learning compatibility, cubic spline interpolation was used to project this variable-density pressure data onto a uniform 100-point grid for both the upper and lower surfaces. This standardization created a consistent 200-point feature vector per flight condition.

## 2. Baseline Surrogate Results
Before implementing physics constraints, two baseline data-driven Multi-Layer Perceptrons (MLPs) were evaluated:
* **Pointwise Approach:** Mapped `[AoA, x/c, surface_flag] ---> Cp`. While mathematically straightforward, treating each point independently stripped away the spatial continuity of the pressure curve.

* **Vector-Output Approach:** Mapped `[AoA] ---> [200-point Cp curve]`. This method successfully maintained the overall geometric shape of the pressure distribution but showed signs of overfitting due to the relatively small CFD sample size.

Additionally, an ensemble of three small surrogate models was trained to predict global forces ($C_l$, $C_d$) directly from the AoA to establish a speed baseline. The ensemble achieved an inference speedup of roughly 246,000x compared to a single ANSYS Fluent run, while maintaining an average full-scale system error of just 0.34% for Lift and 1.16% for Drag.

## 3. PINN Prototype Idea and Preliminary Results
To improve the robustness of the vector-output surrogate, a Physics-Informed Neural Network (PINN) prototype was developed. Instead of relying purely on data-driven Mean Squared Error (MSE), an integral physics constraint was embedded into the training loop.

**The Physics Prior:**
The network was penalized if the area bounded by its predicted $C_p$ curves did not equal the true Lift Coefficient ($C_l$). This was calculated via a custom TensorFlow function performing trapezoidal numerical integration of $(C_{p,lower} - C_{p,upper})$ across the chord. 

**Preliminary Results:**
By combining the scaled data loss with the physical integral loss inside a custom `tf.GradientTape` loop, the network successfully converged. Validation metrics show that while standard data loss stabilized early (expected with limited CFD data), the physics loss plunged by two orders of magnitude. By learning to prioritize physically valid curve geometries that conserve momentum, the PINN prediction perfectly aligned with the true CFD curves during holdout testing.