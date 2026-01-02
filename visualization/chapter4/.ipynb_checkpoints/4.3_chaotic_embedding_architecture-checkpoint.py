#!/usr/bin/env python3
"""
Generate Figure for Section 4.3: Chaotic Embedding Layer Architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.font_manager as fm
from pathlib import Path
current_dir = Path(__file__).parent
font_path = current_dir / '..' / 'Helvetica.ttf'
fm.fontManager.addfont(font_path)

# Set up plotting style - Arial-like font (Helvetica), 11pt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# Set up figure with two subplots: architecture + attractor visualization
fig = plt.figure(figsize=(14, 10))

# Create grid: left side for architecture, right side for attractor
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.05)

# ============ LEFT: Architecture Diagram ============

# Color scheme
color_input = '#E8F4FD'      # Light blue - input
color_mapper = '#FFF3E0'     # Light orange - mapper networks
color_lorenz = '#E8F5E9'     # Light green - Lorenz system
color_output = '#F3E5F5'     # Light purple - output
color_arrow = '#666666'      # Gray arrows
color_text = '#333333'       # Dark text


# ============ RIGHT: Lorenz Attractor Visualization ============
ax2 = fig.add_subplot(gs[1], projection='3d')

# Generate Lorenz attractor trajectory
def lorenz_system(state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def rk4_step(state, dt, sigma=10, rho=28, beta=8/3):
    k1 = lorenz_system(state, sigma, rho, beta)
    k2 = lorenz_system(state + 0.5*dt*k1, sigma, rho, beta)
    k3 = lorenz_system(state + 0.5*dt*k2, sigma, rho, beta)
    k4 = lorenz_system(state + dt*k3, sigma, rho, beta)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Generate trajectory
dt = 0.01
num_steps = 3000
state = np.array([1.0, 1.0, 1.0])

trajectory = np.zeros((num_steps, 3))
trajectory[0] = state

for i in range(1, num_steps):
    state = rk4_step(state, dt)
    trajectory[i] = state

# Generate a second trajectory with slightly different initial condition (different speaker)
state2 = np.array([1.1, 1.0, 1.0])  # Slightly different initial condition
trajectory2 = np.zeros((num_steps, 3))
trajectory2[0] = state2

for i in range(1, num_steps):
    state2 = rk4_step(state2, dt)
    trajectory2[i] = state2

# Plot trajectories
ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
         color='#1976D2', alpha=0.7, linewidth=0.5, label='Speaker A')
ax2.plot(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2],
         color='#D32F2F', alpha=0.7, linewidth=0.5, label='Speaker B')

# Mark initial points
ax2.scatter([1.0], [1.0], [1.0], color='#1976D2', s=50, marker='o', label='Start A')
ax2.scatter([1.1], [1.0], [1.0], color='#D32F2F', s=50, marker='o', label='Start B')

ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.set_zlabel('Z', fontsize=10)
ax2.set_title('Lorenz Attractor\n\n(Different speakers --> Different trajectories)', 
              fontsize=13, fontweight='bold', color='black', pad=10)

# Adjust view angle
ax2.view_init(elev=25, azim=45)

# Add legend
ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)

# Remove grid for cleaner look
ax2.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('chaotic_embedding_architecture.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('chaotic_embedding_architecture.png', 
            dpi=300, bbox_inches='tight', facecolor='white')

print("Figure saved to:")
print("  - chaotic_embedding_architecture.pdf")
print("  - chaotic_embedding_architecture.png")

plt.show()