"""Manuscript figure generation and parameter sweeps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from data_utils import load_emission_rate
from model import simulate


OUTPUT_DIR = Path("figures")

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)


def area_under_curve(values, time):
    """Compute area under a simulated curve."""
    return np.trapz(values, x=time)


def _save_figure(filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")


def _plot_heatmap(
    x_values,
    y_values,
    values,
    x_label,
    y_label,
    colorbar_label,
    title,
    filename,
    contour_color="k",
):
    """Create the repeated parameter-sweep heatmap layout."""
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x_grid, y_grid, values, shading="auto")
    plt.colorbar(label=colorbar_label)

    contour = plt.contour(
        x_grid,
        y_grid,
        values,
        colors=contour_color,
        linewidths=0.5,
    )
    plt.clabel(contour, inline=True, fontsize=8)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, fontsize=12)
    _save_figure(filename)
    plt.show()


def figure_1(emissions=None):
    """Generate the baseline-versus-modified trajectory figure."""
    emissions = load_emission_rate() if emissions is None else emissions

    baseline = simulate(
        0.1, 1, 1, 1, 2, False, emissions
    )
    modified = simulate(
        0.1, 1, 1, 1, 2, True, emissions
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        baseline.time + 1800,
        baseline.temperature,
        label="Baseline Model",
    )
    plt.plot(
        modified.time + 1800,
        modified.temperature,
        label="Modified Model",
    )
    plt.legend(loc="upper left")
    plt.xlabel("Time (year)")
    plt.ylabel("Temperature Anomaly (celsius)")

    axis = plt.gca()
    axis.set_ylim(top=5)

    inset = inset_axes(
        axis,
        width="30%",
        height="30%",
        bbox_to_anchor=(-0.25, 0, 1, 1),
        bbox_transform=axis.transAxes,
        loc="center",
    )
    inset.plot(
        baseline.time + 1800,
        baseline.mitigation_fraction,
    )
    inset.plot(
        modified.time + 1800,
        modified.mitigation_fraction,
    )
    inset.set_xlabel("Time (year)", fontsize=14)
    inset.set_ylabel("X", fontsize=14)

    plt.tight_layout()
    _save_figure("figure_1.png")
    plt.show()


def figure_2_social_learning(emissions=None):
    """Sweep social learning rate and runaway feedback strength."""
    emissions = load_emission_rate() if emissions is None else emissions
    k_values = np.linspace(0, 0.2, 50)
    runaway_values = np.linspace(0, 5, 50)
    values = np.zeros((50, 50))

    for i, runaway_max in enumerate(runaway_values):
        for j, k in enumerate(k_values):
            with_runaway = simulate(
                k, 1, 1, runaway_max, 3, True, emissions
            )
            without_runaway = simulate(
                k, 1, 1, 0, 3, True, emissions
            )
            values[i, j] = (
                area_under_curve(
                    with_runaway.temperature,
                    with_runaway.time,
                )
                - area_under_curve(
                    without_runaway.temperature,
                    without_runaway.time,
                )
            )

    _plot_heatmap(
        k_values,
        runaway_values,
        values,
        "K",
        r"$R_{\mathrm{max}}$",
        "Difference in Area Under the Curve (AUC)",
        "Difference in AUC Vs Runaway Strength and Social Learning Rate",
        "figure_2_social_learning.png",
    )


def figure_2_mitigation_cost(emissions=None):
    """Sweep mitigation cost and runaway feedback strength."""
    emissions = load_emission_rate() if emissions is None else emissions
    beta_values = np.linspace(0, 2, 50)
    runaway_values = np.linspace(0, 5, 50)
    values = np.zeros((50, 50))

    for i, runaway_max in enumerate(runaway_values):
        for j, beta in enumerate(beta_values):
            # The original script omitted its final index argument here.
            # True matches the active-runaway setting used by adjacent sweeps.
            result = simulate(
                0.01,
                beta,
                1,
                runaway_max,
                2,
                True,
                emissions,
            )
            values[i, j] = area_under_curve(
                result.temperature,
                result.time,
            )

    _plot_heatmap(
        beta_values,
        runaway_values,
        values,
        "Beta",
        r"$R_{\mathrm{max}}$",
        "Difference in Area Under the Curve (AUC)",
        "Difference in AUC Vs Runaway Strength and Net Cost of Mitigation",
        "figure_2_mitigation_cost.png",
    )


def figure_2_social_norms(emissions=None):
    """Sweep social norm strength and runaway feedback strength."""
    emissions = load_emission_rate() if emissions is None else emissions
    delta_values = np.linspace(0, 2, 50)
    runaway_values = np.linspace(0, 5, 50)
    values = np.zeros((50, 50))

    for i, runaway_max in enumerate(runaway_values):
        for j, delta in enumerate(delta_values):
            with_runaway = simulate(
                0.01,
                1,
                delta,
                runaway_max,
                3,
                True,
                emissions,
            )
            without_runaway = simulate(
                0.01,
                1,
                delta,
                0,
                3,
                True,
                emissions,
            )
            values[i, j] = (
                area_under_curve(
                    with_runaway.temperature,
                    with_runaway.time,
                )
                - area_under_curve(
                    without_runaway.temperature,
                    without_runaway.time,
                )
            )

    _plot_heatmap(
        delta_values,
        runaway_values,
        values,
        "Delta",
        r"$R_{\mathrm{max}}$",
        "Difference in Area Under the Curve (AUC)",
        "Difference in AUC Vs Runaway Strength and Strength of Social Norms",
        "figure_2_social_norms.png",
    )


def figure_3(emissions=None):
    """Generate the time-to-tipping-point heatmap."""
    emissions = load_emission_rate() if emissions is None else emissions
    k_values = np.linspace(0, 0.2, 50)
    runaway_values = np.linspace(0, 5, 50)
    values = np.full((50, 50), np.nan)

    for i, runaway_max in enumerate(runaway_values):
        for j, k in enumerate(k_values):
            with_runaway = simulate(
                k, 1, 1, runaway_max, 3, True, emissions
            )
            without_runaway = simulate(
                k, 1, 1, 0, 3, True, emissions
            )

            condition = (
                with_runaway.temperature[1:]
                >= 1.1 * without_runaway.temperature[1:]
            )
            if np.any(condition):
                first_index = np.argmax(condition) + 1
                values[i, j] = (
                    with_runaway.time[first_index] + 1800
                )

    x_grid, y_grid = np.meshgrid(k_values, runaway_values)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x_grid, y_grid, values, shading="auto")
    plt.colorbar(label="Time to Tipping Point")
    plt.xlabel("K")
    plt.ylabel(r"$R_{\mathrm{max}}$")
    plt.title(
        "Time to Tipping Point Vs Runaway Strength and Social Learning Rate",
        fontsize=12,
    )
    _save_figure("figure_3.png")
    plt.show()


def figure_4(emissions=None):
    """Generate the peak-temperature heatmap."""
    emissions = load_emission_rate() if emissions is None else emissions
    k_values = np.linspace(0, 0.1, 50)
    runaway_values = np.linspace(0, 5, 50)
    values = np.zeros((50, 50))

    for i, runaway_max in enumerate(runaway_values):
        for j, k in enumerate(k_values):
            result = simulate(
                k, 1, 1, runaway_max, 3, True, emissions
            )
            values[i, j] = np.max(result.temperature)

    _plot_heatmap(
        k_values,
        runaway_values,
        values,
        "K",
        r"$R_{\mathrm{max}}$",
        "Peak of Temperature",
        "Peak Temperature Vs Runaway Strength and Social Learning Rate",
        "figure_4.png",
        contour_color="w",
    )


def figure_5(emissions=None):
    """Sweep social learning rate and critical runaway temperature."""
    emissions = load_emission_rate() if emissions is None else emissions
    k_values = np.linspace(0, 0.1, 50)
    critical_values = np.linspace(1.5, 5, 50)
    values = np.zeros((50, 50))

    for i, critical_temperature in enumerate(critical_values):
        for j, k in enumerate(k_values):
            with_runaway = simulate(
                k,
                1,
                1,
                5,
                critical_temperature,
                True,
                emissions,
            )
            without_runaway = simulate(
                k,
                1,
                1,
                0,
                critical_temperature,
                True,
                emissions,
            )
            values[i, j] = (
                area_under_curve(
                    with_runaway.temperature,
                    with_runaway.time,
                )
                - area_under_curve(
                    without_runaway.temperature,
                    without_runaway.time,
                )
            )

    _plot_heatmap(
        k_values,
        critical_values,
        values,
        "K",
        "Critical Temperature",
        "Difference of Area Under the Curve (AUC)",
        "Difference of AUC Vs Critical Temperature and Social Learning Rate",
        "figure_5.png",
    )


def figure_6(emissions=None):
    """Sweep mitigation cost and runaway feedback strength."""
    emissions = load_emission_rate() if emissions is None else emissions
    beta_values = np.linspace(0, 2, 70)
    runaway_values = np.linspace(0, 6, 70)
    values = np.zeros((70, 70))

    for i, runaway_max in enumerate(runaway_values):
        for j, beta in enumerate(beta_values):
            with_runaway = simulate(
                0.1,
                beta,
                3,
                runaway_max,
                2,
                True,
                emissions,
            )
            without_runaway = simulate(
                0.1,
                beta,
                3,
                0,
                2,
                True,
                emissions,
            )
            values[i, j] = (
                area_under_curve(
                    with_runaway.temperature,
                    with_runaway.time,
                )
                - area_under_curve(
                    without_runaway.temperature,
                    without_runaway.time,
                )
            )

    _plot_heatmap(
        beta_values,
        runaway_values,
        values,
        "Beta",
        r"$R_{\mathrm{max}}$",
        "Difference in Area Under the Curve (AUC)",
        "Difference in AUC Vs Runaway Strength and Net Cost of Mitigation",
        "figure_6.png",
    )


def generate_all_figures():
    """Generate every figure represented in the original script."""
    emissions = load_emission_rate()
    figure_1(emissions)
    figure_2_social_learning(emissions)
    figure_2_mitigation_cost(emissions)
    figure_2_social_norms(emissions)
    figure_3(emissions)
    figure_4(emissions)
    figure_5(emissions)
    figure_6(emissions)
