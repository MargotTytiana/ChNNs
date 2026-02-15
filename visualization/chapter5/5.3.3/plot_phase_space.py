#!/usr/bin/env python3
"""Plot phase space trajectories under clean vs noisy conditions."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')

import matplotlib.font_manager as fm
from pathlib import Path
current_dir = Path(__file__).parent
font_path = current_dir / '..'/ '..' / 'Helvetica.ttf'
fm.fontManager.addfont(font_path)

# Set up matplotlib for academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Helvetica', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
})

def lorenz(state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

def integrate_lorenz(init, dt=0.01, steps=2000):
    traj = np.zeros((steps, 3))
    traj[0] = init
    for i in range(1, steps):
        k1 = lorenz(traj[i-1])
        k2 = lorenz(traj[i-1] + 0.5*dt*k1)
        k3 = lorenz(traj[i-1] + 0.5*dt*k2)
        k4 = lorenz(traj[i-1] + dt*k3)
        traj[i] = traj[i-1] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return traj

# Generate trajectories
np.random.seed(42)
init = np.array([1.0, 1.0, 1.0])
clean = integrate_lorenz(init)
noisy = clean + np.random.randn(*clean.shape) * 3  # Add noise

fig = plt.figure(figsize=(14, 5))

# Clean trajectory
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(clean[:,0], clean[:,1], clean[:,2], color='#662222', lw=0.7, alpha=0.8)
ax1.set_title('Clean Signal', fontsize=12, fontweight='bold')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

# Noisy trajectory
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(noisy[:,0], noisy[:,1], noisy[:,2], color='#7077a1', lw=0.5, alpha=0.6)
ax2.set_title('Noisy Signal (SNR=10dB)', fontsize=12, fontweight='bold')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

# Overlay comparison
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(clean[:,0], clean[:,1], clean[:,2], color='#662222', lw=0.7, alpha=0.8, label='Clean')
ax3.plot(noisy[:,0], noisy[:,1], noisy[:,2], color='#7077a1', lw=0.3, alpha=0.4, label='Noisy')
ax3.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
ax3.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('phase_space_trajectories.png', dpi=300, bbox_inches='tight')
plt.savefig('phase_space_trajectories.pdf', bbox_inches='tight')
print('Saved: phase_space_trajectories.png/pdf')
