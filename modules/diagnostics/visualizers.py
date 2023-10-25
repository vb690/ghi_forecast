import numpy as np

import matplotlib.pyplot as plt

def visualize_forecast_horizon_accuracy(ax, model_name, error_name, 
                                        error_mean, error_sem=None, 
                                        ci=1.96, **plot_kwargs):
    """Visualize the accuracy of a model for each step in the forecast
    horizon. 

    When provided will draw confidence intervals based on 
    the standard error of the mean.
    """
    horizons = np.arange(len(error_mean))

    ax.plot(
        horizons,
        error_mean,
        label=model_name,
        **plot_kwargs
    )

    if error_sem is not None:
        upper_bound = error_mean + (ci * error_sem)
        lower_bound = error_mean - (ci * error_sem)
        ax.fill_between(
            horizons,
            lower_bound,
            upper_bound,
            alpha=0.5
        )
    
    ax.grid()
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel(error_name)

    return ax


def visualize_time_series(ax, time_index, time_series, time_series_name, **plot_kwargs):
    """Visualize q

    When provided will draw confidence intervals based on 
    the standard error of the mean.
    """
    ax.scatter(
        time_index,
        time_series,
        label=time_series_name,
        **plot_kwargs
    )

    ax.tick_params(direction="in", top=True, axis="x", rotation=45)
    ax.grid(
        visible=True, which="major", axis="x", color="k", alpha=0.25, linestyle="--"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    return ax

    
