from datetime import datetime

from tqdm import trange, tqdm

import numpy as np

import pandas as pd


def split_time_series_df(df, start_date, time_column, steps_ahead=None, resolution="m"):
    """Split a time_series dataframe in in-sample and out-sample given a start_date

    If steps_ahehad is provided, the outstample will be limited to t steps_ahead of resolution.
    """
    insample_df = df[df[time_column] < start_date].copy()

    if steps_ahead is not None:
        upper_datetime = insample_df[time_column].max() + pd.to_timedelta(
            steps_ahead, unit=resolution
        )
        outsample_df = df[
            (df[time_column] >= start_date) & (df[time_column] <= upper_datetime)
        ].copy()
    else:
        outsample_df = df[df[time_column] >= start_date].copy()
    return insample_df, outsample_df
