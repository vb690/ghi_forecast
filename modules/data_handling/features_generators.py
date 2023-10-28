import numpy as np


def make_fourier_matrix(time_series_indices, n_components, period):
    """Create a matrix of sine and cosine functions.
    fourier_matrix: matrix of sine and cosine components
    """
    fourier_components = 2 * np.pi * np.arange(1, n_components + 1) / period
    fourier_components = fourier_components * time_series_indices[:, None]
    fourier_matrix = np.concatenate(
        [np.cos(fourier_components), np.sin(fourier_components)], axis=1
    )
    return fourier_matrix
