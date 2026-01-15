# Day 4 – NACA 0012 Mesh Sensitivity Study

## Objective
The objective of Day 4 is to study the effect of mesh resolution on aerodynamic coefficients for a 2D NACA 0012 airfoil using ANSYS Fluent.

---

## Case Setup
- Geometry: NACA 0012 (2D)
- Angle of Attack: 4°
- Reynolds Number: 1 × 10⁶
- Solver: Pressure-based, steady
- Turbulence Model: SST k–ω
- Discretization: Second-order schemes

---

## Mesh Details

| Mesh Type | Cells | Description |
|----------|-------|-------------|
| Coarse | 877 | No boundary-layer refinement |
| Refined | 18,111 | Boundary-layer inflation near airfoil |

---

## Results

| Mesh Type | Cl | Cd |
|----------|----|----|
| Coarse | 0.390 | 0.00164 |
| Refined | 0.404 | 0.01820 |

---

## Discussion
The coarse mesh captures lift reasonably well but severely under-predicts drag due to insufficient near-wall resolution. After refining the mesh and adding boundary-layer elements, the drag coefficient increases significantly while lift changes only marginally. This highlights the strong sensitivity of drag to boundary-layer resolution.

---

## Key Takeaway
Lift is relatively mesh-insensitive, whereas drag prediction strongly depends on near-wall mesh quality. Boundary-layer refinement is essential for accurate aerodynamic drag estimation.
