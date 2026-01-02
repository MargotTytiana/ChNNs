"""
Visualization of Four Chaotic Systems for Speech Modeling
- Lorenz System (3D butterfly attractor)
- Rossler System (3D single spiral attractor)
- Mackey-Glass System (time series + 2D phase space reconstruction)
- Chua's Circuit (3D double-scroll attractor)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.font_manager as fm
from pathlib import Path
current_dir = Path(__file__).parent
font_path = current_dir / '..' / 'Helvetica.ttf'
fm.fontManager.addfont(font_path)

# Set up matplotlib for academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Helvetica', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
})

# Color scheme
TRAJECTORY_COLOR = '#1f77b4'
TRAJECTORY_CMAP = 'viridis'


# =============================================================================
# Lorenz System
# =============================================================================
def lorenz_system(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def generate_lorenz_attractor():
    # Parameters
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    
    # Initial conditions and time span
    state0 = [1.0, 1.0, 1.0]
    t = np.linspace(0, 50, 10000)
    
    # Integrate
    solution = odeint(lorenz_system, state0, t, args=(sigma, rho, beta))
    
    # Discard transient
    transient = 1000
    x, y, z = solution[transient:, 0], solution[transient:, 1], solution[transient:, 2]
    
    return x, y, z


# =============================================================================
# Rossler System
# =============================================================================
def rossler_system(state, t, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]


def generate_rossler_attractor():
    # Parameters
    a, b, c = 0.2, 0.2, 5.7
    
    # Initial conditions and time span
    state0 = [1.0, 1.0, 1.0]
    t = np.linspace(0, 300, 30000)
    
    # Integrate
    solution = odeint(rossler_system, state0, t, args=(a, b, c))
    
    # Discard transient
    transient = 2000
    x, y, z = solution[transient:, 0], solution[transient:, 1], solution[transient:, 2]
    
    return x, y, z


# =============================================================================
# Mackey-Glass System (Delay Differential Equation)
# =============================================================================
def generate_mackey_glass(tau=17, beta=0.2, gamma=0.1, n=10, 
                          duration=1500, dt=0.1):
    """
    Generate Mackey-Glass time series using Euler method.
    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    """
    # Initialize
    history_len = int(tau / dt)
    total_steps = int(duration / dt)
    
    # Initial history (constant)
    x = np.zeros(total_steps + history_len)
    x[:history_len] = 1.2  # Initial condition
    
    # Euler integration
    for i in range(history_len, total_steps + history_len):
        x_tau = x[i - history_len]
        dxdt = beta * x_tau / (1 + x_tau**n) - gamma * x[i-1]
        x[i] = x[i-1] + dxdt * dt
    
    # Discard initial history and transient
    transient = int(500 / dt)
    x = x[history_len + transient:]
    
    return x, tau, dt


# =============================================================================
# Chua's Circuit
# =============================================================================
def chua_circuit(state, t, alpha=10.0, beta=14.87, m0=-1.143, m1=-0.714):
    x, y, z = state
    
    # Piecewise linear function f(x)
    if x > 1:
        fx = m1 * x + (m0 - m1)
    elif x < -1:
        fx = m1 * x - (m0 - m1)
    else:
        fx = m0 * x
    
    dxdt = alpha * (y - x - fx)
    dydt = x - y + z
    dzdt = -beta * y
    
    return [dxdt, dydt, dzdt]


def generate_chua_attractor():
    # Parameters
    alpha, beta = 10.0, 14.87
    m0, m1 = -1.143, -0.714
    
    # Initial conditions and time span
    state0 = [0.1, 0.0, 0.0]
    t = np.linspace(0, 100, 20000)
    
    # Integrate
    solution = odeint(chua_circuit, state0, t, args=(alpha, beta, m0, m1))
    
    # Discard transient
    transient = 2000
    x, y, z = solution[transient:, 0], solution[transient:, 1], solution[transient:, 2]
    
    return x, y, z


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_3d_attractor(ax, x, y, z, title, xlabel='x', ylabel='y', zlabel='z',
                      elev=25, azim=45, use_colormap=True):
    """Plot a 3D attractor with consistent style."""
    if use_colormap:
        # Color by time evolution
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        colors = np.linspace(0, 1, len(x))
        ax.scatter(x, y, z, c=colors, cmap=TRAJECTORY_CMAP, s=0.1, alpha=0.6)
    else:
        ax.plot(x, y, z, color=TRAJECTORY_COLOR, linewidth=0.3, alpha=0.7)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title, y=1)
    ax.view_init(elev=elev, azim=azim)
    
    # Clean up the plot
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')


def plot_mackey_glass_panels(ax1, ax2, x, tau, dt):
    """Plot Mackey-Glass time series and phase space reconstruction."""
    # Time series
    t = np.arange(len(x)) * dt
    ax1.plot(t[:3000], x[:3000], color=TRAJECTORY_COLOR, linewidth=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('x(t)')
    ax1.set_title('(c) Mackey-Glass: Time Series')
    
    # Phase space reconstruction: x(t) vs x(t-tau)
    delay_steps = int(tau / dt)
    x_t = x[delay_steps:]
    x_tau = x[:-delay_steps]
    
    colors = np.linspace(0, 1, len(x_t))
    ax2.scatter(x_tau, x_t, c=colors, cmap=TRAJECTORY_CMAP, s=0.1, alpha=0.5)
    ax2.set_xlabel(r'$x(t-\tau)$')
    ax2.set_ylabel(r'$x(t)$')
    ax2.set_title(r'(d) Mackey-Glass: Phase Space ($\tau=17$)')


def create_combined_figure():
    """Create a 2x2 figure with all four chaotic systems."""
    fig = plt.figure(figsize=(10, 9))
    
    # (a) Lorenz System - 3D
    print("Generating Lorenz attractor...")
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    x, y, z = generate_lorenz_attractor()
    plot_3d_attractor(ax1, x, y, z, '(a) Lorenz Attractor', 
                      elev=20, azim=45)
    
    # (b) Rossler System - 3D
    print("Generating Rossler attractor...")
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    x, y, z = generate_rossler_attractor()
    plot_3d_attractor(ax2, x, y, z, '(b) Rössler Attractor',
                      elev=30, azim=45)
    
    # (c) Mackey-Glass - Time Series
    print("Generating Mackey-Glass system...")
    x_mg, tau, dt = generate_mackey_glass()
    
    ax3 = fig.add_subplot(2, 2, 3)
    t = np.arange(len(x_mg)) * dt
    ax3.plot(t[:3000], x_mg[:3000], color=TRAJECTORY_COLOR, linewidth=0.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('x(t)')
    ax3.set_title('(c) Mackey-Glass: Time Series')
    
    # (d) Mackey-Glass - Phase Space Reconstruction
    ax4 = fig.add_subplot(2, 2, 4)
    delay_steps = int(tau / dt)
    x_t = x_mg[delay_steps:]
    x_tau = x_mg[:-delay_steps]
    colors = np.linspace(0, 1, len(x_t))
    ax4.scatter(x_tau, x_t, c=colors, cmap=TRAJECTORY_CMAP, s=0.1, alpha=0.5)
    ax4.set_xlabel(r'$x(t-\tau)$')
    ax4.set_ylabel(r'$x(t)$')
    ax4.set_title(r'(d) Mackey-Glass: Phase Space ($\tau=17$)')
    
    plt.tight_layout()
    plt.savefig('figure_output/chaotic_systems_overview.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figure_output/chaotic_systems_overview.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figure_output/chaotic_systems_overview.png/pdf")
    plt.show()


def create_lorenz_rossler_figure():
    """Create a figure with Lorenz and Rossler attractors."""
    fig = plt.figure(figsize=(12, 5))
    
    # (a) Lorenz System
    print("Generating Lorenz attractor...")
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x, y, z = generate_lorenz_attractor()
    plot_3d_attractor(ax1, x, y, z, '(a) Lorenz Attractor', 
                      elev=20, azim=45)
    
    # (b) Rossler System
    print("Generating Rossler attractor...")
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    x, y, z = generate_rossler_attractor()
    plot_3d_attractor(ax2, x, y, z, '(b) Rössler Attractor',
                      elev=30, azim=45)
    
    plt.subplots_adjust(wspace=0.15)
    plt.savefig('figure_output/lorenz_rossler_attractors.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figure_output/lorenz_rossler_attractors.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figure_output/lorenz_rossler_attractors.png/pdf")
    plt.close()


def create_mackeyglass_chua_figure():
    """Create a figure with Mackey-Glass and Chua attractors."""
    fig = plt.figure(figsize=(12, 5))
    
    # (a) Mackey-Glass Phase Space (3D embedding)
    print("Generating Mackey-Glass attractor...")
    x_mg, tau, dt = generate_mackey_glass()
    delay_steps = int(tau / dt)
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # 3D time-delay embedding: x(t), x(t-tau), x(t-2*tau)
    x1 = x_mg[2*delay_steps:]
    x2 = x_mg[delay_steps:-delay_steps]
    x3 = x_mg[:-2*delay_steps]
    colors = np.linspace(0, 1, len(x1))
    ax1.scatter(x1, x2, x3, c=colors, cmap=TRAJECTORY_CMAP, s=0.1, alpha=0.5)
    ax1.set_xlabel(r'$x(t)$')
    ax1.set_ylabel(r'$x(t-\tau)$')
    ax1.set_zlabel(r'$x(t-2\tau)$')
    ax1.set_title(r'(a) Mackey-Glass Attractor ($\tau=17$)', y=1)
    ax1.view_init(elev=25, azim=45)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('lightgray')
    ax1.yaxis.pane.set_edgecolor('lightgray')
    ax1.zaxis.pane.set_edgecolor('lightgray')
    
    # (b) Chua's Circuit
    print("Generating Chua attractor...")
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    x, y, z = generate_chua_attractor()
    plot_3d_attractor(ax2, x, y, z, "(b) Chua's Circuit Attractor",
                      elev=25, azim=135)
    
    plt.subplots_adjust(wspace=0.15)
    plt.savefig('figure_output/mackeyglass_chua_attractors.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figure_output/mackeyglass_chua_attractors.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figure_output/mackeyglass_chua_attractors.png/pdf")
    plt.close()


def create_individual_figures():
    """Create individual high-resolution figures for each system."""
    
    # Lorenz
    print("Generating Lorenz attractor...")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = generate_lorenz_attractor()
    plot_3d_attractor(ax, x, y, z, 'Lorenz Attractor', elev=20, azim=45)
    plt.tight_layout()
    plt.savefig('figure_output/lorenz_attractor.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure_output/lorenz_attractor.pdf', dpi=300, bbox_inches='tight',
                facecolor='white')
    print("Saved: figure_output/lorenz_attractor.png/pdf")
    plt.close()
    
    # Rossler
    print("Generating Rossler attractor...")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = generate_rossler_attractor()
    plot_3d_attractor(ax, x, y, z, 'Rössler Attractor', elev=30, azim=45)
    plt.tight_layout()
    plt.savefig('figure_output/rossler_attractor.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure_output/rossler_attractor.pdf', dpi=300, bbox_inches='tight',
                facecolor='white')
    print("Saved: figure_output/rossler_attractor.png/pdf")
    plt.close()
    
    # Mackey-Glass (2-panel)
    print("Generating Mackey-Glass system...")
    x_mg, tau, dt = generate_mackey_glass()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Time series
    t = np.arange(len(x_mg)) * dt
    ax1.plot(t[:3000], x_mg[:3000], color=TRAJECTORY_COLOR, linewidth=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('x(t)')
    ax1.set_title('Time Series')
    
    # Phase space
    delay_steps = int(tau / dt)
    x_t = x_mg[delay_steps:]
    x_tau = x_mg[:-delay_steps]
    colors = np.linspace(0, 1, len(x_t))
    ax2.scatter(x_tau, x_t, c=colors, cmap=TRAJECTORY_CMAP, s=0.1, alpha=0.5)
    ax2.set_xlabel(r'$x(t-\tau)$')
    ax2.set_ylabel(r'$x(t)$')
    ax2.set_title(r'Phase Space Reconstruction ($\tau=17$)')
    
    fig.suptitle('Mackey-Glass System', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('figure_output/mackey_glass_system.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure_output/mackey_glass_system.pdf', dpi=300, bbox_inches='tight',
                facecolor='white')
    print("Saved: figure_output/mackey_glass_system.png/pdf")
    plt.close()
    
    # Chua
    print("Generating Chua attractor...")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = generate_chua_attractor()
    plot_3d_attractor(ax, x, y, z, "Chua's Circuit Attractor", elev=25, azim=135)
    plt.tight_layout()
    plt.savefig('figure_output/chua_attractor.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('figure_output/chua_attractor.pdf', dpi=300, bbox_inches='tight',
                facecolor='white')
    print("Saved: figure_output/chua_attractor.png/pdf")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("Chaotic Systems Visualization")
    print("=" * 60)
    
    # Create two separate figures
    create_lorenz_rossler_figure()
    create_mackeyglass_chua_figure()
    
    print("\nDone!")