"""The SelectionExperiment module.

Encapsulates methods to store, persist and visualize the utility and data quality results from data summarization.
Assumes that one can have several subset selection `methods` and each method is evaluated across different `metrics`.
"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import FHP_COLORMAP, FHP_MARKER_TYPE, data_processed_dir, report_dir
from loguru import logger


class SelectionMethod:
    """Class representing a single coreset selection method and its metrics."""

    def __init__(self, method_name):
        self.method_name = method_name
        self.metrics = dict.fromkeys(
            [
                "Fraction",
                "Accuracy",
                "Precision",
                "Recall",
                "GIQA",
                "FID",
                "MSID",
                "Coverage",
                "Density",
            ],
            None,
        )

    def set_metric(self, metric_name, values):
        """Set values for a specific metric."""
        if metric_name not in self.metrics:
            raise ValueError(f"Invalid metric name: {metric_name}")
        self.metrics[metric_name] = np.array(values)

    def get_metric(self, metric_name):
        """Get values for a specific metric."""
        if metric_name not in self.metrics:
            raise ValueError(f"Invalid metric name: {metric_name}")
        return self.metrics[metric_name]

    def to_dataframe(self):
        """Convert all metrics to a DataFrame with 'Fraction' as the index."""
        data = {metric: values for metric, values in self.metrics.items() if values is not None}
        df = pd.DataFrame(data)

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
            experiment = pickle.load(f)  # nosec

        return experiment

    def plot_all_metrics(self, figsize=(24, 4), save_path=None):
        """Plot all metrics in a single row of subplots.

        Each method is represented by a different colored line with
        unique cap style within each subplot.
        """
        methods, metrics = self.get_method_names(), self.get_metric_names()

        if not methods or not metrics:
            logger.error("No data available to plot.")
            return None, None

        fig, axes = plt.subplots(1, len(metrics) - 1, figsize=figsize)
        axes = np.array(axes).flatten()

        for j, metric_name in enumerate(metrics[1:]):
            ax = axes[j]
            for i, method_name in enumerate(methods):
                method = self.methods[method_name]
                metric_data = method.get_metric(metric_name)

                if metric_data is not None:
                    ax.plot(
                        method.get_metric("Fraction"),
                        metric_data,
                        marker=FHP_MARKER_TYPE[i],
                        color=FHP_COLORMAP[i],
                        label=method_name,
                    )

            ax.set_title(metric_name, fontsize=12)
            ax.set_xlabel("Subset size / %")
            ax.set_xlim(-25, 125)

            if j == 0:
                ax.set_ylabel("Value")
                ax.legend(loc="lower right", framealpha=0, fontsize=9)

        plt.tight_layout()

        if save_path:
            fname = Path(os.path.join(report_dir, save_path))
            plt.savefig(fname, dpi=300, bbox_inches="tight")

        return fig, axes
