from datetime import datetime

from tqdm import trange

import numpy as np

import pandas as pd



def split_time_series_df(df, start_date, time_column, steps_ahead=None, resolution="m"):
    """Split a time_series dataframe in in-sample and out-sample given a start_date

    If steps_ahehad is provided, the outstample will be limited to t steps_ahead of resolution.
    """
    insample_df = df[df[time_column] < start_date].copy()

    if steps_ahead is not None:
        upper_datetime = insample_df[time_column].max() + pd.to_timedelta(steps_ahead, unit=resolution)
        outsample_df = df[(df[time_column] >= start_date) & (df[time_column] <= upper_datetime)].copy()
    else:
         df[df[time_column] >= start_date].copy()
    return insample_df, outsample_df


def expanding_split_time_series_df(df, start_date, time_column, frequency, steps_ahead, resolution="m", early_stop=None):
    """Split a time_series dataframe in in-sample and out-sample given a start_date generating an 
    expanding window.

    """
    start_date = pd.to_datetime(start_date)
    early_stop_counter = 0
    while start_date < df[time_column] - pd.Timedelta(steps_ahead, unit=resolution):

        insample_df, outsample_df = split_time_series_df(
            df=df,
            start_date=start_date,
            time_column=time_column,
            steps_ahead=steps_ahead,
            resolution=resolution
        )

        start_date += pd.to_timedelta(frequency, unit=resolution)
        early_stop_counter += 1
        if early_stop_counter > early_stop:
            break
        yield insample_df, outsample_df


def tabularize_time_series(time_series, insample_window, forecast_window):
    """Tabularize a univariate time series generating and in-sample and out-sample 
    arrays of size in_sample_window and forecast_window respectively
    """
    final_idx = len(time_series) - (forecast_window + insample_window)
    insample_series = []
    outsample_series = []
    for start_idx in trange(final_idx):

        end_insample_idx = start_idx + insample_window
        end_outsample_idx = end_insample_idx + forecast_window

        insample_series.append(
            time_series[start_idx : end_insample_idx]
        )
        outsample_series.append(
            time_series[end_insample_idx : end_outsample_idx]
        )
    
    insample_series = np.array(insample_series)
    outsample_series = np.array(outsample_series)
    return insample_series, outsample_series

