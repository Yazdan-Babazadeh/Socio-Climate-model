# Socio-Climate System Simulation and Visualization

This repository contains the simulation code and visualization scripts from the manuscript:

**"Social dynamics can delay or prevent climate tipping points by speeding the adoption of climate change mitigation."**  
_Proceedings A, 2025_

## Overview

This code models the coupled dynamics between:
- Atmospheric carbon and other carbon reservoirs
- Global temperature
- Human mitigation behavior influenced by social learning, norms, and perceived risk

The model simulates the interactions through a system of ODEs and generates visualizations for manuscript figures. 

## Features

- Simulates socio-climate dynamics using `scipy.integrate.solve_ivp`
- Includes climate feedback mechanisms (vegetation decay, soil respiration, ocean uptake)
- Models human behavior through imitation dynamics and a logistic risk perception model
- Produces figures corresponding to the main and supplementary panels of the manuscript

## Dependencies

pip install numpy pandas matplotlib scipy

File Structure
global.1751_2017.csv: Historical emissions data (global fossil fuel CO₂ emissions)

main.py (or your script name): Contains all simulation logic and plotting code

Model Components
State variables:
C_a (atmospheric carbon), C_o (ocean carbon), C_v (vegetation carbon),
C_so (soil carbon), T (temperature anomaly), x (mitigation adoption)

Functions:

simulate(k, beta, delta, r_max, T_critical, index): Runs the ODE solver

AUC(Y, X): Computes area under the curve for comparisons

Plotting functions generate panels for the manuscript

Behavioral Dynamics:

Social learning rate: k

Net mitigation cost: beta

Strength of social norms: delta

Perceived runaway tipping risk: controlled via r_max, T_critical

How to Use
Run Simulations and Generate All Figures:
Simply execute the script from your terminal:

bash
Copy
Edit
python main.py
Customize Simulations:
You can change parameters (e.g., k, beta, delta, r_max) directly in the plotting functions or loops to explore alternative dynamics.

Outputs:

.png images for all figures will be saved automatically (e.g., figure_for_paper.png)

Plots include time evolution of variables and contour plots comparing intervention impacts

Figures Generated
Figure 1: Comparison of baseline vs modified socio-climate dynamics

Figure 2: Panels A–F — Sensitivity analysis for k, beta, delta, and r_max

Figure 3: Time to tipping point

Figure 4: Peak temperature reached

Figure 5: Impact of varying social learning and temperature threshold

Figure 6: Net mitigation cost vs runaway feedback strength

Notes
The simulation assumes time is discretized in 0.01-year steps for smooth plotting

The emissions data is trimmed and scaled to align with model assumptions

Each simulation runs for 400 years (from year 1800 to 2200)

License
This code is for academic and research use. Please cite the associated paper if used in any derivative work.
