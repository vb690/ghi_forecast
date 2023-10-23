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