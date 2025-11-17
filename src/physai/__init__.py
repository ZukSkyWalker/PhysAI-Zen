# physai/__init__.py

from .ising_data import generate_ising_samples, IsingDataset
from .ising_transformer import IsingTransformer
from .ising_diagnostics import (
    validate_ising_samples,
    print_validation_report,
    compute_magnetization,
    compute_correlation_function,
    theoretical_correlation_1d,
)
from .training import train_ising_transformer, plot_training_curve

__version__ = "0.1.0"

__all__ = [
    "generate_ising_samples",
    "IsingDataset",
    "IsingTransformer",
    "validate_ising_samples",
    "print_validation_report",
    "compute_magnetization",
    "compute_correlation_function",
    "theoretical_correlation_1d",
    "train_ising_transformer",
    "plot_training_curve",
]
