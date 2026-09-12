# Socio-Climate System Simulation and Visualization

This repository contains the simulation and visualization code associated with:

**"Social dynamics can delay or prevent climate tipping points by speeding the adoption of climate change mitigation."**  
*Proceedings A, 2025*

## Overview

The model couples atmospheric, ocean, vegetation, and soil carbon reservoirs with global temperature and mitigation adoption. Human behavior is influenced by social learning, mitigation cost, social norms, and perceived climate risk.

The dynamical system is integrated with scipy.integrate.solve_ivp using the BDF method.

## Repository structure

- **main.py** — command-line entry point for manuscript figures.
- **model.py** — coupled six-state ODE model and simulation routine.
- **parameters.py** — physical, carbon-cycle, climate, and behavioral constants.
- **data_utils.py** — historical emissions loading and preprocessing.
- **figures.py** — manuscript figures and parameter sweeps.
- **global.1751_2017.csv** — historical global fossil-fuel CO2 emissions data.
- **requirements.txt** — Python dependencies.

The original single-file implementation, Code.py, remains available in the Git history. The last commit containing that original root-level implementation is:

d7ee3d9fae3334e93fc1ccc4715d0c57740bf1fa — **Update README.md**

## State variables

The six model states are:

1. Atmospheric carbon, C_a
2. Ocean carbon, C_o
3. Vegetation carbon, C_v
4. Soil carbon, C_so
5. Temperature anomaly, T
6. Mitigation adoption fraction, x

The current code unpacks these states by name inside the derivative function instead of repeatedly using numeric array indices.

## Installation

~~~bash
pip install -r requirements.txt
~~~

## Running the model

Generate all figures represented in the original script:

~~~bash
python main.py
~~~

Generate a specific figure or parameter-sweep panel:

~~~bash
python main.py --figure 1
python main.py --figure 2-k
python main.py --figure 2-beta
python main.py --figure 2-delta
python main.py --figure 3
python main.py --figure 4
python main.py --figure 5
python main.py --figure 6
~~~

Generated images are saved in the figures directory.

## Figure mapping

- **Figure 1** — baseline versus modified socio-climate trajectories, with mitigation adoption inset.
- **Figure 2-k** — AUC sensitivity to social learning rate k and runaway feedback strength.
- **Figure 2-beta** — sensitivity to mitigation cost beta and runaway feedback strength.
- **Figure 2-delta** — AUC sensitivity to social norm strength delta and runaway feedback strength.
- **Figure 3** — time to a 10% temperature divergence.
- **Figure 4** — peak temperature over social learning and runaway feedback strength.
- **Figure 5** — AUC difference over social learning and critical runaway temperature.
- **Figure 6** — AUC difference over mitigation cost and runaway feedback strength.

## Stage-1 cleanup

This version is a code-quality and reproducibility cleanup. It intentionally does **not** change the governing equations, parameter values, historical-emissions transformation, initial conditions, BDF solver choice, post-historical emissions formula, or parameter-sweep ranges.

Changes include:

- separating model equations, parameters, data loading, plotting, and execution;
- removing repeated imports and repeated AUC definitions;
- removing unused variables and stale commented exploratory code;
- replacing positional simulation returns with named result fields;
- documenting model terms and state variables;
- fixing stale attempts to unpack eleven values from a seven-value simulation return;
- supplying the missing runaway-mode argument in the mitigation-cost sweep using the active-runaway setting used by the surrounding modified-model sweeps;
- computing Figure 3 tipping time directly from the solver time array rather than assuming exactly 100 array indices per year.

Potential methodological changes — including changing the emissions interpolation, tipping criterion, AUC definitions, runaway-feedback formulation, initial conditions, or solver configuration — are intentionally outside this cleanup.

## Notes

- Historical emissions remain annual piecewise-constant forcing, matching the original implementation.
- Each default simulation spans 400 model years with 100 requested output points per year.
- The post-historical emissions extension is preserved exactly from the original model.

## Citation

If you use this repository in derivative academic work, please cite the associated paper.
