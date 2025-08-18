import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_samples = 5  # Number of signals
time_steps = 100  # Time steps T
amplitude = 5  # Maximum amplitude of signals

# Generate random signals (samples x time steps)
signals = np.random.randn(num_samples, time_steps) * amplitude

# Create a figure with subplots arranged vertically
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(num_samples, 1, figure=fig, hspace=0.3)

# Plot each signal
for i in range(num_samples):
    ax = fig.add_subplot(gs[i, 0])

    # Set transparency: first signal is fully opaque, others are transparent
    alpha = 1.0 if i == 0 else 0.4

    # Plot the signal
    ax.plot(signals[i], color='blue', alpha=alpha, linewidth=2)

    # Add title for each subplot
    ax.set_title(f'Signal {i + 1}', fontsize=10)

    # Customize axes
    ax.set_xlim(0, time_steps)
    ax.set_ylim(-amplitude * 1.2, amplitude * 1.2)
    ax.set_xticks([])  # Remove x-ticks for cleaner look
    if i == num_samples - 1:  # Only show x-label for the last plot
        ax.set_xlabel('Time Steps (T)', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)

# Add an overall title
fig.suptitle('Random Signals', fontsize=14, y=0.95)

# Adjust layout and display
plt.tight_layout()
plt.show()
