import numpy as np

from scipy.stats import sem


def compute_rmse(y_true, y_forecast, axis=None):
    """Compute the root mean squared error for a series
    of steps_ahead forecasts.

    The functions expects ground truth and forecast in a
    window X steps_ahead format.
    """
    squared_error = (y_true - y_forecast) ** 2
    mean_squared_error = np.mean(squared_error, axis=axis)
    sem_squared_error = sem(squared_error, axis=axis)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    root_sem_squared_error = np.sqrt(sem_squared_error)
    return root_mean_squared_error, root_sem_squared_error


def compute_symmetric_absolute_percentage_error(y, yhat):
    """Compute the symmetric version of the absolute percentage error
    between ground truth values and predictions. It is less biased than
    the absolute_percentage_error and bounded between 0 and 100.
    """
    sape = (np.abs(y - yhat) / (np.abs(y) + np.abs(yhat))) * 100
    return sape
