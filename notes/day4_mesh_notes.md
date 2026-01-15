# Day 4 – Mesh Sensitivity Notes (NACA 0012)

## Context
This day focused on understanding how mesh resolution affects aerodynamic results for a 2D NACA 0012 airfoil in ANSYS Fluent.

Case details:
- AoA: 4°
- Reynolds number: 1e6
- Solver: steady, pressure-based
- Turbulence model: SST k–ω

---

## Key Observations

### 1. Lift converges early
- Coarse mesh (877 cells) already produced a reasonable lift coefficient (~0.39).
- Refining the mesh only changed Cl by a few percent (~0.404).
- This confirms that lift is primarily pressure-dominated and less sensitive to near-wall resolution.

### 2. Drag is extremely mesh-sensitive
- Coarse mesh severely under-predicted drag (Cd ≈ 0.0016).
- After adding boundary-layer refinement and increasing cell count (~18k cells), Cd increased to ≈ 0.018.
- This large jump is due to improved wall shear stress resolution.

### 3. Boundary layers are non-negotiable for drag
- Without inflation layers, Fluent cannot resolve velocity gradients near the wall.
- Wall shear stress collapses on coarse meshes, leading to unrealistically low drag.
- High-aspect-ratio cells near the wall are expected and acceptable.

---

## Numerical / Solver Insights
- Residual convergence alone is not sufficient; force coefficients must be monitored.
- Both meshes converged numerically, but only the refined mesh produced physically meaningful drag.
- Increased mesh resolution increased runtime but improved accuracy.

---

## Reference Values Pitfall
- Incorrect reference values initially caused inconsistent Cl/Cd.
- Reference velocity, area, and depth must be set explicitly for 2D cases.
- Report definitions must be recreated after changing reference values.

---

## Day 4 Takeaway
Lift converges quickly on coarse meshes, but accurate drag prediction requires proper boundary-layer resolution.
