import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

import seaborn as sns


def visualize_forecast_horizon_accuracy(
    ax, model_name, error_name, error_mean, error_sem=None, ci=1.96, **plot_kwargs
):
    """Visualize the accuracy of a model for each step in the forecast
    horizon.

    When provided will draw confidence intervals based on
    the standard error of the mean.
    """
    horizons = np.arange(len(error_mean))

    ax.plot(horizons, error_mean, label=model_name, **plot_kwargs)

    if error_sem is not None:
        upper_bound = error_mean + (ci * error_sem)
        lower_bound = error_mean - (ci * error_sem)
        ax.fill_between(horizons, lower_bound, upper_bound, alpha=0.25)

    ax.grid()
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel(error_name)

    return ax


def visualize_time_series(ax, time_series, time_series_name, **plot_kwargs):
    """Visualize q

    When provided will draw confidence intervals based on
    the standard error of the mean.
    """
    ax.scatter(
        time_series.index, 
        time_series.values, 
        label=time_series_name, 
        **plot_kwargs,
    )

    ax.tick_params(direction="in", top=True, axis="x", rotation=45)
    ax.grid(
        visible=True, which="major", axis="x", color="k", alpha=0.25, linestyle="--"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    return ax


def visualize_time_series_components_performance(
    time_series_train,
    time_series_test,
    components_insample,
    components_outsample,
    target_name,
    model_name,
    link_function,
    guardrail_metric,
    loss,
    history,
    figsize=(15, 5),
):
    """Visulize the perfomance of a forecast given its components
    """
    forecast_insample = link_function(sum(components_insample.values()))
    forecast_outsample = link_function(sum(components_outsample.values()))

    residuals_insample = time_series_train.values - forecast_insample
    residuals_outsample = time_series_test.values - forecast_outsample

    computed_guardrail_metric = guardrail_metric(
        y=time_series_test.values, 
        yhat=forecast_outsample,
    )
    med_guardrail_metric = round(np.median(computed_guardrail_metric), 3)

    fig = plt.figure(tight_layout=True, figsize=figsize)
    columns = max(4, len(components_insample) + 1)
    gs = gridspec.GridSpec(2, columns, figure=fig)

    ax_time_series = fig.add_subplot(gs[0, 2:])
    ax_loss_history = fig.add_subplot(gs[0, 1])
    ax_performance = fig.add_subplot(gs[0, 0])

    ax_time_series.scatter(
        time_series_test.index,
        time_series_test.values,
        s=0.5,
        c="k",
        label="Ground Truth",
    )
    ax_time_series.scatter(
        time_series_train.index,
        time_series_train.values,
        s=0.5,
        c="k",
    )

    ax_time_series.plot(
        time_series_train.index,
        forecast_insample,
        c="tab:blue",
        label="In-Sample Forecast",
    )
    ax_time_series.plot(
        time_series_test.index,
        forecast_outsample,
        linestyle="--",
        c="tab:blue",
        label="Out-Sample Forecast",
    )
    ax_time_series.set_ylabel(target_name)
    ax_time_series.set_xlabel("Date")

    ax_time_series.tick_params(direction="in", top=True, axis="x")
    ax_time_series.grid(
        visible=True, 
        which="major", 
        axis="x", 
        color="k", 
        alpha=0.25, 
        linestyle="--",
    )
    ax_time_series.legend()

    sns.kdeplot(computed_guardrail_metric, ax=ax_performance, clip=(0, 100))
    ax_performance.axvline(
        med_guardrail_metric, 
        linestyle=":", 
        c="r", 
        label=f"Median \n {med_guardrail_metric}",
    )
    ax_performance.set_title("Test Guardrail Metric")
    ax_performance.set_xlabel(r"$100 \times (\frac{|y - \hat{y}|}{|y| + |\hat{y}|})$")
    ax_performance.set_ylabel("Density")
    ax_performance.legend()

    ax_loss_history.plot(
        np.arange(len(history)),
        history,
    )
    ax_loss_history.set_title("Training Loss")
    ax_loss_history.set_ylabel(loss)
    ax_loss_history.set_xlabel("Epoch")

    ax_time_series.axvline(time_series_train.index.max(), c="r", linestyle="--")

    for column, component_name in enumerate(components_insample.keys()):
        ax_component = fig.add_subplot(gs[1, column])

        component_insample = components_insample[component_name]
        component_outsample = components_outsample[component_name]

        ax_component.plot(
            time_series_train.index,
            component_insample,
            c="tab:blue",
            label="In-Sample",
        )
        ax_component.plot(
            time_series_test.index,
            component_outsample,
            linestyle="--",
            c="tab:blue",
            label="Out-Sample",
        )

        ax_component.axvline(time_series_train.index.max(), c="r", linestyle="--")
        ax_component.tick_params(direction="in", top=True, axis="x")
        ax_component.set_title(component_name.capitalize())
        ax_component.grid(
            visible=True, 
            which="major", 
            axis="x", 
            color="k", 
            alpha=0.25, 
            linestyle="--",
        )
        ax_component.set_ylabel(target_name)
        ax_component.set_xlabel("Date")

    ax_residuals = fig.add_subplot(gs[1, -1])
    ax_residuals.plot(
        time_series_train.index,
        residuals_insample,
        c="tab:blue",
        label="In-Sample",
    )
    ax_residuals.plot(
        time_series_test.index,
        residuals_outsample,
        linestyle="--",
        c="tab:blue",
        label="Out-Sample",
    )

    ax_residuals.axvline(time_series_train.index.max(), c="r", linestyle="--")
    ax_residuals.axhline(0, c="k", linestyle=":")
    ax_residuals.tick_params(direction="in", top=True, axis="x")
    ax_residuals.set_title("Residuals")
    ax_residuals.grid(
        visible=True, 
        which="major", 
        axis="x", 
        color="k", 
        alpha=0.25, 
        linestyle="--",
    )
    ax_residuals.set_ylabel(target_name)
    ax_residuals.set_xlabel("Date")
    
    plt.suptitle(f"Model {model_name}")
    plt.tight_layout()
    return fig
