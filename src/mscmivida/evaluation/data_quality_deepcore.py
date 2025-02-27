from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from loguru import logger


def parse_experiment_checkpoint(path: str) -> tuple[float, float]:
    """Parses experiment checkpoint filename to extract fraction and
    accuracy."""
    try:
        parts = Path(path).stem.split("_")
        accuracy = float(parts[-1][:5])
        fraction = float(parts[-2]) * 100
        return fraction, accuracy
    except (IndexError, ValueError) as e:
        logger.error(f"Failed to parse checkpoint path: {path} | Error: {e}")
        return 0.0, 0.0


evaluation_metrics_db: dict[str, list] = {"Fraction": [], "Accuracy": []}

CKPT_FOLDER = Path("/home/duarte.folgado/TeslaVM/DeepCore/result/")
for f in CKPT_FOLDER.rglob("*.ckpt"):
    fraction, accuracy = parse_experiment_checkpoint(str(f))
    evaluation_metrics_db["Fraction"].append(fraction)
    evaluation_metrics_db["Accuracy"].append(accuracy)

# TODO: Add data quality methods

evaluation_metrics_db = pd.DataFrame(evaluation_metrics_db)

# Plot Fraction vs Accuracy
fig, ax = plt.subplots()
ax.plot(evaluation_metrics_db["Fraction"], evaluation_metrics_db["Accuracy"], "o-")
ax.set_xlabel("Fraction", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

plt.show(block=False)
