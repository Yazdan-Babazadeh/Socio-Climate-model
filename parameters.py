"""Model constants for the coupled socio-climate system.

Values are preserved from the original research script. This module only
centralizes them so the governing equations are easier to read and audit.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParameters:
    """Physical, carbon-cycle, climate, and behavioral model constants."""

    soil_carbon_0: float = 1500
    vegetation_carbon_0: float = 550
    atmospheric_carbon_0: float = 596
    ocean_carbon_0: float = 1.5e5

    f_gtm: float = 8.3259e13
    k_a: float = 1.773e20
    k_p: float = 0.184
    k_area: float = 8.7039e9
    k_mm: float = 1.478
    k_c: float = 29e-6
    k_m: float = 120e-6
    k_r: float = 0.092
    activation_energy: float = 54.83
    gas_constant: float = 8.314
    k_soil_respiration: float = 0.034
    k_b: float = 157.072
    k_turnover: float = 0.092
    ocean_flux_0: float = 2.5e-2
    ocean_mixing_fraction: float = 0.3
    zeta: float = 50

    reference_temperature: float = 288.15
    relative_humidity: float = 0.5915
    reference_pressure: float = 1.4e11
    latent_heat: float = 43655
    albedo: float = 0.225
    solar_constant: float = 1368
    heat_capacity: float = 4.69e23
    stefan_boltzmann: float = 5.67e-8
    earth_area: float = 5.101e14

    perceived_risk_max: float = 5
    perceived_risk_steepness: float = 3
    perceived_risk_midpoint: float = 1.5
    runaway_steepness: float = 5

    future_emissions_max: float = 6
    future_emissions_saturation: float = 50


PARAMS = ModelParameters()
