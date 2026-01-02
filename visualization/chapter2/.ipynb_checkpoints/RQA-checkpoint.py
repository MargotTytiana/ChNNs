import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse


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

# ============================================================
# Generate signals: periodic (sine) vs chaotic (Lorenz)
# ============================================================

# Periodic signal (sine wave)
t_periodic = np.linspace(0, 4*np.pi, 300)
signal_periodic = np.sin(t_periodic)

# Chaotic signal (Lorenz system, x-component)
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

sol = solve_ivp(lorenz, (0, 30), [1.0, 1.0, 1.0], 
                t_eval=np.linspace(0, 30, 300), method='RK45')
signal_chaotic = sol.y[0]

# ============================================================
# Phase space reconstruction (time-delay embedding)
# ============================================================
def embed_signal(signal, delay=5, dim=3):
    n = len(signal) - (dim - 1) * delay
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = signal[i * delay : i * delay + n]
    return embedded

# ============================================================
# Construct recurrence matrix
# ============================================================
def recurrence_matrix(embedded, threshold_percentile=15):
    dist_matrix = squareform(pdist(embedded, metric='euclidean'))
    threshold = np.percentile(dist_matrix[dist_matrix > 0], threshold_percentile)
    R = (dist_matrix <= threshold).astype(int)
    return R

# Embed and compute recurrence matrices
emb_periodic = embed_signal(signal_periodic, delay=8, dim=3)
emb_chaotic = embed_signal(signal_chaotic, delay=3, dim=3)

R_periodic = recurrence_matrix(emb_periodic, threshold_percentile=10)
R_chaotic = recurrence_matrix(emb_chaotic, threshold_percentile=10)

# ============================================================
# Figure: Comparison of Recurrence Plots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

# Top-left: Periodic signal
axes[0, 0].plot(t_periodic[:len(emb_periodic)], signal_periodic[:len(emb_periodic)], 
                '#2B5F8C', linewidth=1.8)
axes[0, 0].set_xlabel('Time', fontsize=10)
axes[0, 0].set_ylabel('Amplitude', fontsize=10)
axes[0, 0].set_title('(a) Periodic Signal (Sine Wave)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

rp_cmap = ListedColormap(['white', '#444143']) 
# Top-right: Recurrence plot of periodic signal
axes[0, 1].imshow(R_periodic, cmap=rp_cmap, origin='lower', aspect='equal')
axes[0, 1].set_xlabel('Time index $i$', fontsize=10)
axes[0, 1].set_ylabel('Time index $j$', fontsize=10)
axes[0, 1].set_title('(b) Recurrence Plot: Periodic', fontsize=11)

# Add annotation for diagonal lines
axes[0, 1].annotate('Diagonal lines\n(high DET)', 
                    xy=(180, 180), xytext=(200, 135),
                    fontsize=9, color='#529AD9',
                    arrowprops=dict(arrowstyle='->', color='#529AD9', lw=1.5))

# Bottom-left: Chaotic signal
t_chaotic = np.linspace(0, 30, len(emb_chaotic))
axes[1, 0].plot(t_chaotic, signal_chaotic[:len(emb_chaotic)], 
                '#3C7DA6', linewidth=1.2)
axes[1, 0].set_xlabel('Time', fontsize=10)
axes[1, 0].set_ylabel('$x(t)$', fontsize=10)
axes[1, 0].set_title('(c) Chaotic Signal (Lorenz $x$-component)', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

rb_cmap = ListedColormap(['white', '#444143']) 
# Bottom-right: Recurrence plot of chaotic signal
axes[1, 1].imshow(R_chaotic, cmap=rb_cmap, origin='lower', aspect='equal')
axes[1, 1].set_xlabel('Time index $i$', fontsize=10)
axes[1, 1].set_ylabel('Time index $j$', fontsize=10)
axes[1, 1].set_title('(d) Recurrence Plot: Chaotic', fontsize=11)

# Add annotations for chaotic RP structures
axes[1, 1].annotate('Diagonal segments\n(determinism)', 
                    xy=(205, 210), xytext=(225, 160),
                    fontsize=9, color='#529AD9',
                    arrowprops=dict(arrowstyle='->', color='#529AD9', lw=1.5))
axes[1, 1].annotate('Vertical lines\n(laminar states)', 
                    xy=(122, 60), xytext=(130, 23),
                    fontsize=9, color='green',
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.0))
ellipse = Ellipse(
    (50, 60),
    width=175,
    height=140,
    angle=45,
    edgecolor='green',
    facecolor='none',
    linestyle='--',
    linewidth=1.5,
    hatch='..',
    clip_on=False,
    alpha=0.35
)
axes[1, 1].add_patch(ellipse)

plt.tight_layout()
# plt.subplots_adjust(
#     left=0.08,
#     right=0.85,
#     top=0.95,
#     bottom=0.08,
#     wspace=0.25,
#     hspace=0.30
# )
plt.savefig('rqa_comparison.png', dpi=300)
plt.savefig('rqa_comparison.pdf')
plt.show()

print("Figure saved as rqa_comparison.png and rqa_comparison.pdf")