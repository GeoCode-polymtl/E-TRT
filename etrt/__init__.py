"""
E-TRT: Electrical Thermal Response Test simulation and inversion.

This package provides tools for simulating and inverting electrical thermal
response test (E-TRT) data. It includes thermal models (ILS, ICS), ERT
simulation capabilities, Bayesian inversion methods, and visualization tools.

Main modules:
    lib_etrt: Core functionality for thermal models and ERT simulation
    grid_fun: Plotting and visualization utilities for grid search results
"""

from .lib_etrt import (
    ILS,
    ICS,
    dt2sigma,
    ert_setup,
    simulate_ert,
    simulate_etrt,
    jacobian_ert_fd,
    bayesian_inversion,
    CTi,
    get_Cdi,
    print_matrix,
)
from .grid_fun import (
    plot_sum_across_slices,
    plot_prob,
    plot_grid_field,
)

__version__ = "0.1.0"
__author__ = "Clarissa Szabo-Som and Gabriel Fabien-Ouellet"

__all__ = [
    # Thermal models
    "ILS",
    "ICS",
    # Utilities
    "dt2sigma",
    "ert_setup",
    "print_matrix",
    # Simulation
    "simulate_ert",
    "simulate_etrt",
    "jacobian_ert_fd",
    # Inversion
    "bayesian_inversion",
    "CTi",
    "get_Cdi",
    # Visualization
    "plot_sum_across_slices",
    "plot_prob",
    "plot_grid_field",
]

