import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from data.loader_mitbih import MITBIHDataset

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

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

    axs[i].plot(np.linspace(0, 500, len(mean_heartbeat)), mean_heartbeat, color="k", label="Mean")

    axs[i].fill_between(
        np.linspace(0, 500, len(mean_heartbeat)),
        mean_heartbeat - std_heartbeat,
        mean_heartbeat + std_heartbeat,
        color="gray",
        alpha=0.3,
        label="±1 Std Dev",
    )

    axs[i].set_ylim(-2, 2.8)
    axs[i].set_title(HEARTBEAT_CLASS_MAPPING.get(label, "Unknown"))

plt.tight_layout()
plt.show(block=False)
plt.savefig("Meanwaves.pdf")

# A short block of code to generate representative heartbeat traces for the intuition image.
# 6x6. let us consider a scenario with 80% of heartbeats from healthy patients and 20% with SVEB
# Create a dataset of representative traces
# Constants. Use seed 521752 and 8784651
NUMBER_OF_TRACES = 36
HEALTHY_RATIO = 0.8
UNHEALTHY_RATIO = 1 - HEALTHY_RATIO
RANDOM_SEED = 8784651
np.random.seed(RANDOM_SEED)

# Select representative indices
healthy_indices = np.where(y == 0)[0]
unhealthy_indices = np.where(y == 2)[0]

healthy_representative_idx = np.random.choice(
    healthy_indices,
    int(np.ceil(NUMBER_OF_TRACES * HEALTHY_RATIO)),
    replace=False,
)
unhealthy_representative_idx = np.random.choice(
    unhealthy_indices,
    int(np.floor(NUMBER_OF_TRACES * UNHEALTHY_RATIO)),
    replace=False,
)

representative_idx = np.concatenate([healthy_representative_idx, unhealthy_representative_idx])
np.random.shuffle(representative_idx)  # Optional: Shuffle the indices for variety in display

# Plot representative heartbeat traces
fig, ax = plt.subplots(6, 6, figsize=(12, 12))
axes = ax.flatten()  # Flatten axes once for efficiency

for i, idx in enumerate(representative_idx):
    axes[i].plot(X[idx], lw=1.5)  # Adjust line width for better visualization
    axes[i].set_title(HEARTBEAT_CLASS_MAPPING[y[idx]], fontsize=8)
    axes[i].axis("off")  # Turn off axis for cleaner visualization

# Despine and adjust layout
for unused_axis in axes[len(representative_idx) :]:  # Hide unused subplots
    unused_axis.axis("off")

plt.tight_layout()
plt.show(block=False)
