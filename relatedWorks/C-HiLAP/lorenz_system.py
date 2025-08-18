import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# 1. 3D Visualization of Lorenz Attractor (core of chaotic dynamics)
def lorenz_system(t, state, sigma=10, rho=28, beta=8/3, Wx_h=0):
    x, y, z = state
    dxdt = sigma * (y - x) + Wx_h  # With neural network coupling term
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Solve Lorenz equations
initial_state = [1.0, 1.0, 1.0]  # Initial state
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000)
solution = solve_ivp(
    lorenz_system, t_span, initial_state, t_eval=t_eval,
    args=(10, 28, 8/3, 0)  # Standard parameters, no coupling term
)

# Plot Lorenz attractor (3D)
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(solution.y[0], solution.y[1], solution.y[2], color='blue', linewidth=0.5)
ax1.set_title('Lorenz Attractor (Chaotic State)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# 2. 3D Phase Space Reconstruction of Speech Signal
# Generate simulated speech signal (sinusoid with noise)
t = np.linspace(0, 20, 1000)
voice_signal = np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.random.randn(len(t))  # 1D speech signal

# Phase space reconstruction parameters (Takens' theorem)
tau = 20  # Time delay (autocorrelation zero-crossing)
d_e = 3   # Embedding dimension (for 3D visualization)
n = len(voice_signal) - 2 * tau  # Number of data points after reconstruction

# Construct 3D phase space
X = voice_signal[:n]
Y = voice_signal[tau:tau+n]
Z = voice_signal[2*tau:2*tau+n]

# Plot 3D phase space
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(X, Y, Z, c=t[:n], cmap='viridis', s=1)
ax2.set_title('3D Phase Space Reconstruction of Speech Signal')
ax2.set_xlabel('s(t)')
ax2.set_ylabel('s(t+τ)')
ax2.set_zlabel('s(t+2τ)')

# 3. Illustration of topological features for attractor pooling (basis for correlation dimension calculation)
# Randomly sample points from Lorenz attractor to simulate input to attractor pooling layer
attractor_points = np.array([solution.y[0][::100], solution.y[1][::100], solution.y[2][::100]]).T  # Downsampled

# Plot attractor point cloud (used for correlation dimension calculation)
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(attractor_points[:,0], attractor_points[:,1], attractor_points[:,2],
           color='red', alpha=0.6, s=5)
ax3.set_title('3D Point Cloud for Attractor Pooling Layer')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.text2D(0.05, 0.95, "Used for calculating correlation dimension D₂", transform=ax3.transAxes, fontsize=8)

plt.tight_layout()
plt.show()
