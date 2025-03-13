import os

import matplotlib.pyplot as plt
from config import data_processed_dir
from evaluation.selection_experiment import SelectionExperiment

if __name__ == "__main__":
    EXPERIMENT_PATH = os.path.join(data_processed_dir, "deepcore_methods_experiment.pkl")
    experiment = SelectionExperiment().load(EXPERIMENT_PATH)

    # Figure 1: Comparison between different selection methods across different data quality metrics.
    experiment.plot_all_metrics()
    plt.show(block=False)
