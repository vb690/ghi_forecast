import numpy as np

import pandas as pd


def static_stat_forecast(time_series, steps_ahead, stat_func):
    """Generate a steps_ahead forecast propagating the last value
    of a statistic computed on the time series.
    """
    computed_stat = stat_func(time_series)
    length_time_series = len(time_series)
    in_sample_forecast = np.full(length_time_series, computed_stat)
    out_of_sample_forecast = np.full(steps_ahead, computed_stat)
    return in_sample_forecast, out_of_sample_forecast


def ewm_forecast(time_series, steps_ahead, alpha=1, **ewm_kwargs):
    """Generate a steps_ahead forecast propagating the last
    value of an expontial weighted mean of the time_series.

    By setting alpha to 1 we obtain a peristence model.
    """
    in_sample_forecast = (
        pd.Series(time_series).ewm(alpha=alpha, **ewm_kwargs).mean().values
    )
    last_value = in_sample_forecast[-1]
    out_of_sample_forecast = np.full(steps_ahead, last_value)
    return in_sample_forecast, out_of_sample_forecast
