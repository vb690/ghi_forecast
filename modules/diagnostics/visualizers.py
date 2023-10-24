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

    
