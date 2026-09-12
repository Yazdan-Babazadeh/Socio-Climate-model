"""Coupled socio-climate ODE model.

The equations and numerical settings are preserved from the original
research script. The main changes are organization, naming, documentation,
and returning a structured simulation result.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from data_utils import load_emission_rate
from parameters import PARAMS


@dataclass
class SimulationResult:
    """Time series returned by one socio-climate simulation."""

    time: np.ndarray
    atmospheric_carbon: np.ndarray
    ocean_carbon: np.ndarray
    vegetation_carbon: np.ndarray
    soil_carbon: np.ndarray
    temperature: np.ndarray
    mitigation_fraction: np.ndarray


def simulate(
    social_learning_rate,
    mitigation_cost,
    social_norm_strength,
    runaway_max,
    critical_temperature,
    runaway_enabled,
    emissions=None,
    parameters=PARAMS,
    simulation_time=400,
    points_per_year=100,
):
    """Integrate the six-state socio-climate model."""
    p = parameters
    emission_rate = load_emission_rate() if emissions is None else emissions

    def emissions_forcing(time):
        # The original model uses annual, piecewise-constant forcing.
        year_index = int(time)
        historical_end = 216

        if year_index < historical_end:
            return emission_rate[year_index]

        elapsed = year_index - historical_end
        extension = (
            elapsed * p.future_emissions_max
            / (elapsed + p.future_emissions_saturation)
        )
        return extension + emission_rate[historical_end]

    def runaway_feedback(temperature):
        if not runaway_enabled:
            return 0.0
        return runaway_max / (
            1
            + np.exp(
                -p.runaway_steepness
                * (temperature - critical_temperature)
            )
        )

    def co2_partial_pressure(atmospheric_carbon):
        return (
            p.f_gtm
            * (atmospheric_carbon + p.atmospheric_carbon_0)
            / p.k_a
        )

    def photosynthesis(atmospheric_carbon, temperature):
        pressure = co2_partial_pressure(atmospheric_carbon)
        co2_response = (pressure - p.k_c) / (
            p.k_m + pressure - p.k_c
        )
        temperature_response = (
            ((15 + temperature) ** 2) * (25 - temperature) / 5625
        )

        if pressure - p.k_c > 0 and -15 < temperature < 25:
            return (
                p.k_p
                * p.vegetation_carbon_0
                * p.k_mm
                * co2_response
                * temperature_response
            )
        return 0.0

    def vegetation_respiration(vegetation_carbon, temperature):
        return (
            p.k_r
            * vegetation_carbon
            * p.k_area
            * np.exp(
                -p.activation_energy
                / (p.gas_constant * (temperature + p.reference_temperature))
            )
        )

    def soil_respiration(temperature, soil_carbon):
        return (
            p.k_soil_respiration
            * soil_carbon
            * p.k_b
            * np.exp(
                -308.56
                / (temperature + p.reference_temperature - 227.13)
            )
        )

    def litter_flux(vegetation_carbon):
        return p.k_turnover * vegetation_carbon

    def ocean_flux(atmospheric_carbon, ocean_carbon):
        return (
            p.ocean_flux_0
            * p.ocean_mixing_fraction
            * (
                atmospheric_carbon
                - p.zeta
                * p.atmospheric_carbon_0
                * ocean_carbon
                / p.ocean_carbon_0
            )
        )

    def optical_depth(atmospheric_carbon, temperature):
        co2_term = 1.73 * co2_partial_pressure(atmospheric_carbon) ** 0.263
        water_vapor_term = 0.0126 * (
            p.relative_humidity
            * p.reference_pressure
            * np.exp(
                -p.latent_heat
                / (
                    p.gas_constant
                    * (temperature + p.reference_temperature)
                )
            )
        ) ** 0.503
        return co2_term + water_vapor_term + 0.0231

    def incoming_radiation(atmospheric_carbon, temperature):
        absorbed_solar = (1 - p.albedo) * p.solar_constant / 4
        greenhouse_factor = 1 + 0.75 * optical_depth(
            atmospheric_carbon,
            temperature,
        )
        return absorbed_solar * greenhouse_factor

    def perceived_climate_risk(temperature):
        return p.perceived_risk_max / (
            1
            + np.exp(
                -p.perceived_risk_steepness
                * (temperature - p.perceived_risk_midpoint)
            )
        )

    def derivatives(time, state):
        (
            atmospheric_carbon,
            ocean_carbon,
            vegetation_carbon,
            soil_carbon,
            temperature,
            mitigation_fraction,
        ) = state

        photosynthesis_rate = photosynthesis(
            atmospheric_carbon,
            temperature,
        )
        vegetation_respiration_rate = vegetation_respiration(
            vegetation_carbon,
            temperature,
        )
        soil_respiration_rate = soil_respiration(
            temperature,
            soil_carbon,
        )
        ocean_flux_rate = ocean_flux(
            atmospheric_carbon,
            ocean_carbon,
        )
        litter_rate = litter_flux(vegetation_carbon)

        d_atmosphere = (
            emissions_forcing(time) * (1 - mitigation_fraction)
            - photosynthesis_rate
            + vegetation_respiration_rate
            + soil_respiration_rate
            - ocean_flux_rate
            + runaway_feedback(temperature)
        )

        d_ocean = ocean_flux_rate
        d_vegetation = (
            photosynthesis_rate
            - vegetation_respiration_rate
            - litter_rate
        )
        d_soil = litter_rate - soil_respiration_rate

        energy_imbalance = (
            incoming_radiation(atmospheric_carbon, temperature)
            - p.stefan_boltzmann
            * (temperature + p.reference_temperature) ** 4
        )
        d_temperature = (
            p.earth_area
            / p.heat_capacity
            * energy_imbalance
            * 3.14e7
        )

        if time < 216:
            d_mitigation = 0.0
        else:
            d_mitigation = (
                social_learning_rate
                * mitigation_fraction
                * (1 - mitigation_fraction)
                * (
                    -mitigation_cost
                    + perceived_climate_risk(temperature)
                    + social_norm_strength
                    * (2 * mitigation_fraction - 1)
                )
            )

        return np.array(
            [
                d_atmosphere,
                d_ocean,
                d_vegetation,
                d_soil,
                d_temperature,
                d_mitigation,
            ]
        )

    initial_state = np.array([0, 0, 0, 0, 0, 0.05], dtype=float)
    evaluation_times = np.linspace(
        0,
        simulation_time,
        simulation_time * points_per_year,
    )

    solution = solve_ivp(
        derivatives,
        (0, simulation_time),
        initial_state,
        method="BDF",
        t_eval=evaluation_times,
    )

    return SimulationResult(
        time=solution.t,
        atmospheric_carbon=solution.y[0],
        ocean_carbon=solution.y[1],
        vegetation_carbon=solution.y[2],
        soil_carbon=solution.y[3],
        temperature=solution.y[4],
        mitigation_fraction=solution.y[5],
    )
