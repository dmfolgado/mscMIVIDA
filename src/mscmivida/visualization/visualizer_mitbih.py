import matplotlib.pyplot as plt
import numpy as np
from data.loader_mitbih import MITBIHDataset

HEARTBEAT_CLASS_MAPPING = {
    0: "Normal",
    1: "Ventricular ectopic beat",
    2: "Supraventricular ectopic beat",
    3: "Fusion beat",
    4: "Unknown beat",
}

X, y, s = MITBIHDataset(subsets="aami").load_data()

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

for i, label in enumerate(sorted(set(y))):
    indices = np.where(y == label)[0]
    mean_heartbeat = np.mean(X[indices], axis=0)
    std_heartbeat = np.std(X[indices], axis=0)

    axs[i].plot(mean_heartbeat, color="k", label="Mean")

    axs[i].fill_between(
        range(len(mean_heartbeat)),
        mean_heartbeat - std_heartbeat,
        mean_heartbeat + std_heartbeat,
        color="gray",
        alpha=0.3,
        label="±1 Std Dev",
    )

    axs[i].set_title(HEARTBEAT_CLASS_MAPPING.get(label, "Unknown"))

plt.tight_layout()
plt.show(block=False)
