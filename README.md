# 📘 Numerical Methods for Partial Differential Equations

This repository showcases two major projects developed during the course *Numerical Methods for PDEs*, aiming to solve and analyze various physical problems using the Finite Element Method (FEM). The objective is to demonstrate how numerical tools and adaptive strategies can be implemented in practice to achieve accurate and stable solutions for elliptic, parabolic, and fluid flow problems.

---

## 📂 Main Projects

### 🔹 H1_PolNavarroPerez.pdf — Laplace Equation & Porous Media Flow

This report is divided into two parts:

- **Laplace equation in 2D**:  
  We solve the Laplace equation on a square domain using triangular (P1) and quadrilateral (Q1, Q2) finite elements. The convergence of the method is validated by computing the \( L^2 \) and \( H^1 \) errors as the element size is reduced, and the slope of convergence is compared to theoretical predictions.

- **Porous media simulation**:  
  The flow of water through a 2D orthotropic porous medium is modeled using Darcy’s law. Dirichlet boundary conditions are imposed both via system reduction and Lagrange multipliers. The piezometric level and total water flow across the excavation boundary are computed for different mesh configurations and domain extensions.

---

### 🔹 H2_PolNavarroPerez.pdf — Gradient Smoothing, Heat Equation & Navier-Stokes

This second report covers three distinct problems:

1. **Gradient smoothing & ZZ error estimation**:  
   For a 1D Poisson problem, the smoothed gradient \( q \) is computed as an \( L^2 \) projection of the FEM derivative, enabling accurate error estimation via the Zienkiewicz-Zhu (ZZ) indicator. Adaptive mesh refinement is implemented based on local error density.

2. **Transient heat equation**:  
   A thin film is heated over time, and its temperature evolution is modeled with the heat equation. Both explicit and implicit Euler time integration schemes are implemented, along with a stability analysis to choose the appropriate time step.

3. **Cavity flow (Navier–Stokes)**:  
   Incompressible 2D cavity flow is simulated using a Q2-Q1 FEM pair and Picard iterations. The code handles increasing Reynolds numbers (up to Re ≈ 3560) with an incremental strategy to ensure stability. Flow patterns and secondary vortex formation are clearly observed through streamline visualizations.

---

## 💻 Code and Structure

The code is written in **Python**, using structured scripts and modules for:
- Mesh generation and reference elements
- System assembly using Gauss quadrature
- Boundary condition handling (including Dirichlet and Lagrange multipliers)
- Error computation in \( L^2 \) and \( H^1 \) norms
- Adaptive refinement and Picard iterations
- Visualization with `matplotlib` and `mpl_toolkits.mplot3d`

Each major assignment or section is organized inside the `assignments/` folder. Final reports are included in `final_report/`, and additional code is found under `codes_H1/` and `codes_H2/`.

---

## 📦 Dependencies

This project uses basic scientific Python libraries. A minimal list includes:

```bash
numpy
scipy
matplotlib

