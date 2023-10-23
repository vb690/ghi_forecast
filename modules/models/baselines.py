import numpy as np

import pandas as pd


def static_stat_forecast(time_series, steps_ahead, stat_func):
    """Generate a steps_ahead forecast propagating the last value
    of a statistic computed on the time series.
    """
    computed_stat = stat_func(time_series)
    length_time_series = len(time_series)
    in_sample_forecast = np.array([computed_stat] * length_time_series)
    out_of_sample_forecast = np.array([computed_stat] * steps_ahead)
    return in_sample_forecast, out_of_sample_forecast

def rolling_stat_forecast(time_series, steps_ahead, stat_func, window_size, **rolling_kwargs):
    """Generate a steps_ahead forecast propagating the last value
    of a rolling statistic computed on the time series.
    """
    rolling_stat = pd.Series(time_series).rolling(window=window_size, **rolling_kwargs).apply(stat_func)
    in_sample_forecast = rolling_stat.values
    last_value = in_sample_forecast[-1]
    out_of_sample_forecast = np.array([last_value] * steps_ahead)
    return in_sample_forecast, out_of_sample_forecast

def ewm_forecast(time_series, steps_ahead, alpha=1, **ewm_kwargs):
    """Generate a steps_ahead forecast propagating the last
    value of an expontial weighted mean of the time_series.

    By setting alpha to 1 we obtain a peristence model.
    """
    in_sample_forecast = pd.Series(time_series).ewm(alpha=alpha, **ewm_kwargs).mean().values
    last_value = in_sample_forecast[-1]
    out_of_sample_forecast = np.array([last_value] * steps_ahead)
    return in_sample_forecast, out_of_sample_forecast
