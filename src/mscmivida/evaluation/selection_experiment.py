"""The SelectionExperiment module.

Encapsulates methods to store, persist and visualize the utility and data quality results from data summarization.
Assumes that one can have several subset selection `methods` and each method is evaluated across different `metrics`.
"""

import os
import pickle
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mscmivida.config import data_processed_dir, report_dir
from loguru import logger


class SelectionMethod:
    """Class representing a single coreset selection method and its metrics."""

    def __init__(self, method_name):
        self.method_name = method_name
        self.metrics = {}
        #self.metric_names = ["metric1", "metric2", "metric3", "metric4"]  # TODO: Replace for the actual metric names.
        self.metric_names = ["Fraction", "Accuracy", "Precision", "Recall", "GIQA", "FID", "MSID", "Coverage", "Density"]

        for metric in self.metric_names:
            self.metrics[metric] = None

    def set_metric(self, metric_name, values):
        """Set values for a specific metric."""
        if metric_name not in self.metric_names:
            raise ValueError(f"Invalid metric name: {metric_name}")

        self.metrics[metric_name] = np.array(values)

    def get_metric(self, metric_name):
        """Get values for a specific metric."""
        if metric_name not in self.metric_names:
            raise ValueError(f"Invalid metric name: {metric_name}")

        return self.metrics[metric_name]

    def to_dataframe(self):
        """Convert all metrics to a DataFrame for analysis."""
        df = pd.DataFrame()
        for metric in self.metric_names:
            if self.metrics[metric] is not None:
                df[metric] = self.metrics[metric]
        return df


class SelectionExperiment:
    """Class to manage multiple coreset selection methods."""

    def __init__(self, experiment_name="coreset_experiment"):
        self.methods = {}
        self.method_names = []
        self.experiment_name = experiment_name

    def add_method(self, method_name):
        """Add a new coreset selection method to the experiment."""
        if method_name not in self.methods:
            method = SelectionMethod(method_name)
            self.methods[method_name] = method
            self.method_names.append(method_name)
            return method
        else:
            return self.methods[method_name]

    def get_method_names(self):
        """Get list of all method names in the experiment."""
        return self.method_names

    def get_metric_names(self):
        """Get list of metric names used in the experiment."""
        if len(self.method_names) > 0:
            first_method = self.methods[self.method_names[0]]
            return first_method.metric_names
        return []

    def compare_methods(self, metric_name):
        """Compare all methods for a specific metric."""
        comparison = pd.DataFrame()

        for name in self.method_names:
            method = self.methods[name]
            metric_data = method.get_metric(metric_name)
            if metric_data is not None:
                comparison[name] = metric_data

        return comparison

    def get_all_results(self):
        """Get a comprehensive DataFrame with all results."""
        results_dict = {}

        for method_name in self.method_names:
            method = self.methods[method_name]
            for metric_name in method.metric_names:
                if method.metrics[metric_name] is not None:
                    column_name = (method_name, metric_name)
                    results_dict[column_name] = method.metrics[metric_name]

        all_data = pd.DataFrame({k: pd.Series(v) for k, v in results_dict.items()})

        if not all_data.empty:
            all_data.columns = pd.MultiIndex.from_tuples(
                all_data.columns,
                names=["Method", "Metric"],
            )

        return all_data

    def save(self, directory=data_processed_dir):
        """Save the experiment to disk using pickle."""
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        experiment_path = os.path.join(directory, f"{self.experiment_name}.pkl")
        with open(experiment_path, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Experiment saved to {experiment_path}")

    @classmethod
    def load(cls, experiment_path):
        """Load an experiment from disk using pickle."""
        with open(experiment_path, "rb") as f:
            experiment = pickle.load(f)

        return experiment

    def plot_all_metrics(self, figsize=(18, 6), save_path=None):
        """Plot all metrics in a single row of subplots.

        Each method is represented by a different colored line with
        unique cap style within each subplot.
        """
        methods = self.get_method_names()
        metrics = self.get_metric_names()

        if not methods or not metrics:
            logger.error("No data to plot")
            return None, None

        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

        if len(metrics) == 1:
            axes = np.array([axes])

        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        markers = cycle(["o", "s", "^", "D", "v", "p", "*", "h", "X"])
        linestyles = cycle(["-", "--", "-.", ":"])

        for j, metric_name in enumerate(metrics):
            ax = axes[j]

            for i, method_name in enumerate(methods):
                method = self.methods[method_name]
                metric_data = method.get_metric(metric_name)

                if metric_data is not None:
                    x = np.arange(len(metric_data))

                    color = colors[i]
                    marker = next(markers)
                    linestyle = next(linestyles)

                    ax.plot(
                        x,
                        metric_data,
                        marker=marker,
                        linestyle=linestyle,
                        color=color,
                        markeredgecolor="black",
                        markeredgewidth=1.0,
                        label=method_name,
                    )

            ax.set_title(metric_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Index")

            if j == 0:
                ax.set_ylabel("Value")

            if j == 0:
                ax.legend(loc="best", frameon=True, framealpha=0.9, title="Methods")

        plt.suptitle("Comparison of all metrics across methods", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(Path(os.path.join(report_dir, save_path)), dpi=300, bbox_inches="tight")

        return fig, axes

    def plot_metric_comparison(self, metric_name, figsize=(10, 6), save_path=None):
        """Plot a comparison of all methods for a single metric with enhanced
        styling."""
        data = {}
        for method_name in self.method_names:
            method = self.methods[method_name]
            metric_data = method.get_metric(metric_name)
            if metric_data is not None:
                data[method_name] = metric_data

        if not data:
            logger.error(f"No data available for metric: {metric_name}")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
        markers = cycle(["o", "s", "^", "D", "v", "p", "*", "h", "X"])
        linestyles = cycle(["-", "--", "-.", ":"])

        for i, (method_name, values) in enumerate(data.items()):
            x = np.arange(len(values))

            color = colors[i]
            marker = next(markers)
            linestyle = next(linestyles)

            ax.plot(
                x,
                values,
                marker=marker,
                linestyle=linestyle,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1.0,
                label=method_name,
            )

        ax.set_title(f"Comparison of {metric_name} across methods", fontsize=14, fontweight="bold")
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.legend(loc="best", frameon=True, framealpha=0.9, title="Methods")

        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(report_dir, save_path), dpi=300, bbox_inches="tight")

        return fig, ax


if __name__ == "__main__":
    # Create experiment
    experiment = SelectionExperiment(experiment_name="enhanced_plots_experiment")

    # List of the CoreSet selection methods. Try to crawl the folder names.
    method_list = ["Method_A", "Method_B", "Method_C", "Method_D"]

    # Add all methods and set example data with different array sizes
    for method_name in method_list:
        method = experiment.add_method(method_name)

        # Generate some example data with different patterns for visualization
        for i, metric_name in enumerate(method.metric_names):
            # Create data with different patterns
            if method_name == "Method_A":
                x = np.linspace(0, 10, 15)
                values = np.sin(x) + i + np.random.randn(15) * 0.3
            elif method_name == "Method_B":
                x = np.linspace(0, 10, 15)
                values = x**0.5 + i + np.random.randn(15) * 0.3
            elif method_name == "Method_C":
                x = np.linspace(0, 10, 15)
                values = np.cos(x) + i * 0.5 + np.random.randn(15) * 0.3
            else:
                x = np.linspace(0, 10, 15)
                values = np.log(x + 1) + i * 0.8 + np.random.randn(15) * 0.3

            method.set_metric(metric_name, values)

    # Saves
    experiment.save()

    # Plot all metrics
    _, _ = experiment.plot_all_metrics(figsize=(20, 5), save_path="all_metrics.png")

    # Plot comparison for a specific metric
    _, _ = experiment.plot_metric_comparison("metric1", save_path="specific_metric.png")

    plt.show(block=False)
