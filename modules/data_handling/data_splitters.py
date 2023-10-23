from datetime import datetime

import pandas as pd



def split_time_series_df(df, start_date, time_column, steps_ahead, resolution):
    insample_df = df[df[time_column] < start_date].copy()

    upper_datetime = insample_df[time_column].max() + pd.to_timedelta(steps_ahead, unit=resolution)
    outsample_df = df[(df[time_column] >= start_date) & (df[time_column] <= upper_datetime)]
    return insample_df, outsample_df