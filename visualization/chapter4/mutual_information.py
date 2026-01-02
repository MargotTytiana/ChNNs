import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from pathlib import Path

current_dir = Path(__file__).parent
font_path = current_dir / '..' / 'Helvetica.ttf'
if font_path.exists():
    fm.fontManager.addfont(str(font_path))

# Set academic plot style
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Helvetica', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.0,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# ============================================
# PART 1: Generate and plot the original signal
# ============================================

# Generate the signal
t = np.linspace(0, 8*np.pi, 200)
wave = np.sin(t) * (1 + 0.3*np.sin(0.5*t))

fig1, ax = plt.subplots(1, 1, figsize=(16, 3))
ax.plot(t, wave, color='#3498DB', linewidth=1.5)
ax.axis('off')

# Save in both formats
fig1.savefig('signal_waveform.pdf', format='pdf')
fig1.savefig('signal_waveform.png', format='png', dpi=300)

print("Signal waveform saved as:")
print("  - signal_waveform.pdf")
print("  - signal_waveform.png")

# ============================================
# PART 2: Mutual Information and FNN Analysis
# (Aligned with precompute_embedding_params.py)
# ============================================

def compute_mutual_information(signal: np.ndarray, max_delay: int = 50, n_bins: int = 50) -> np.ndarray:
    """
    Compute mutual information for different time lags.
    Matches the implementation in precompute_embedding_params.py.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input time series
    max_delay : int
        Maximum delay to compute
    n_bins : int
        Number of histogram bins (default: 50, same as precompute script)
    
    Returns:
    --------
    mi_values : np.ndarray
        Mutual information values for tau = 1 to max_delay
    """
    # Limit signal length for efficiency (same as precompute script)
    max_len = 10000
    if len(signal) > max_len:
        signal = signal[:max_len]
    
    mi_values = np.zeros(max_delay)
    
    for tau in range(1, max_delay + 1):
        x1 = signal[:-tau]
        x2 = signal[tau:]
        
        # Compute 2D histogram
        hist_2d, _, _ = np.histogram2d(x1, x2, bins=n_bins)
        
        # Normalize to get joint probability
        p_joint = hist_2d / hist_2d.sum()
        
        # Marginal probabilities
        p_x1 = p_joint.sum(axis=1)
        p_x2 = p_joint.sum(axis=0)
        
        # Compute mutual information using log2 (same as precompute script)
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_joint[i, j] > 1e-10 and p_x1[i] > 1e-10 and p_x2[j] > 1e-10:
                    mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_x1[i] * p_x2[j]))
        
        mi_values[tau - 1] = mi
    
    return mi_values


def find_first_minimum(mi_values: np.ndarray) -> int:
    """
    Find first local minimum of mutual information.
    Matches the implementation in precompute_embedding_params.py.
    
    Returns:
    --------
    tau : int
        Optimal delay (1-indexed)
    """
    # Find first local minimum (same logic as precompute script)
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
            return i + 1  # Return 1-indexed tau
    
    # Fallback: return point of maximum decrease
    diff = np.diff(mi_values)
    return np.argmin(diff) + 2


def compute_fnn(signal: np.ndarray, tau: int, max_dim: int = 15, 
                threshold: float = 0.05, rtol: float = 15.0) -> np.ndarray:
    """
    Compute false nearest neighbors percentage for different dimensions.
    Matches the implementation in precompute_embedding_params.py.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input time series
    tau : int
        Time delay for embedding
    max_dim : int
        Maximum embedding dimension to test
    threshold : float
        FNN fraction threshold to determine optimal dimension
    rtol : float
        Relative tolerance threshold for FNN criterion
    
    Returns:
    --------
    fnn_percentages : np.ndarray
        Percentage of false nearest neighbors for each dimension
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Limit signal length (same as precompute script)
    max_len = 5000
    if len(signal) > max_len:
        signal = signal[:max_len]
    
    fnn_percentages = np.zeros(max_dim)
    fnn_percentages[0] = 100.0  # d=1 cannot be tested
    
    for dim in range(2, max_dim + 1):
        n_points = len(signal) - dim * tau
        if n_points < 100:
            fnn_percentages[dim - 1] = fnn_percentages[dim - 2] if dim > 1 else 100.0
            continue
        
        # Create embedding for dim and dim+1
        embedded_d = np.zeros((n_points, dim))
        embedded_d1 = np.zeros((n_points, dim + 1))
        
        for i in range(dim):
            embedded_d[:, i] = signal[i * tau:i * tau + n_points]
        for i in range(dim + 1):
            embedded_d1[:, i] = signal[i * tau:i * tau + n_points]
        
        # Find nearest neighbors in d-dimensional space
        nbrs = NearestNeighbors(n_neighbors=2).fit(embedded_d)
        distances, indices = nbrs.kneighbors(embedded_d)
        
        # Count false neighbors
        n_false = 0
        n_test = min(500, n_points)
        test_indices = np.random.choice(n_points, n_test, replace=False)
        
        for idx in test_indices:
            nn_idx = indices[idx, 1]  # Nearest neighbor index
            
            # Distance in d-space
            dist_d = distances[idx, 1]
            if dist_d < 1e-10:
                continue
            
            # Distance in (d+1)-space
            dist_d1 = np.linalg.norm(embedded_d1[idx] - embedded_d1[nn_idx])
            
            # Check if false neighbor (same criterion as precompute script)
            if (dist_d1 - dist_d) / dist_d > rtol:
                n_false += 1
        
        fnn_percentages[dim - 1] = (n_false / n_test) * 100
    
    return fnn_percentages


def find_optimal_dimension(fnn_values: np.ndarray, threshold_percent: float = 5.0) -> int:
    """
    Find optimal embedding dimension where FNN drops below threshold.
    
    Parameters:
    -----------
    fnn_values : np.ndarray
        FNN percentages for each dimension
    threshold_percent : float
        Threshold percentage (default: 5%)
    
    Returns:
    --------
    optimal_dim : int
        Optimal embedding dimension (1-indexed)
    """
    fnn_below = np.where(fnn_values < threshold_percent)[0]
    return fnn_below[0] + 1 if len(fnn_below) > 0 else len(fnn_values)


# ============================================
# Compute the analyses
# ============================================

max_lag = 40
mi_values = compute_mutual_information(wave, max_delay=max_lag)

# Find first minimum of MI
mi_first_min = find_first_minimum(mi_values)

# Compute FNN with the recommended tau
max_dim = 8
np.random.seed(42)  # For reproducibility
fnn_values = compute_fnn(wave, tau=mi_first_min, max_dim=max_dim)

# Find optimal dimension
optimal_dim = find_optimal_dimension(fnn_values, threshold_percent=5.0)

# ============================================
# Create the combined figure with two subplots
# ============================================

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left subplot: Mutual Information Function
lags = np.arange(1, max_lag + 1)
ax2.plot(lags, mi_values, 'o-', color='#E15759', linewidth=2.0, markersize=5)
ax2.axvline(x=mi_first_min, color='#2E86AB', linestyle='--', 
            linewidth=1.5, alpha=0.8, label=f'First minimum: $\\tau={mi_first_min}$')
ax2.plot(mi_first_min, mi_values[mi_first_min - 1], 'o', markersize=11,
         markerfacecolor='none', markeredgecolor='#2E86AB', markeredgewidth=2)

ax2.set_xlabel('Time Lag $\\tau$ (samples)')
ax2.set_ylabel('Mutual Information $I(\\tau)$ (bits)')
ax2.set_title('Mutual Information Function', pad=12)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.set_xlim(0, max_lag)
ax2.set_ylim(0, np.max(mi_values) * 1.1)

# Right subplot: FNN Percentage
dims = np.arange(1, max_dim + 1)
ax3.plot(dims, fnn_values, 's-', color='#59A14F', linewidth=2.0, markersize=7)
ax3.axhline(y=5, color='#666666', linestyle=':', linewidth=1.5, alpha=0.7, label='5% threshold')

# Mark the optimal dimension if found
if optimal_dim <= max_dim:
    ax3.plot(optimal_dim, fnn_values[optimal_dim - 1], 'o', markersize=11, 
             markerfacecolor='none', markeredgecolor='#E15759', markeredgewidth=2)
    
    # Annotate the optimal dimension
    ax3.annotate(f'Optimal: $d={optimal_dim}$', 
                 xy=(optimal_dim, fnn_values[optimal_dim - 1]),
                 xytext=(optimal_dim + 0.5, fnn_values[optimal_dim - 1] + 15),
                 arrowprops=dict(arrowstyle='->', color='#E15759', alpha=0.7),
                 fontsize=10, fontweight='bold', color='#E15759')

ax3.set_xlabel('Embedding Dimension $d$')
ax3.set_ylabel('False Nearest Neighbors (%)')
ax3.set_title('False Nearest Neighbors Analysis', pad=12)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper right', framealpha=0.9)
ax3.set_xlim(0.5, max_dim + 0.5)
ax3.set_ylim(0, max(100, np.max(fnn_values) * 1.1))

# Adjust layout
plt.tight_layout()

# Save the second figure in both formats
fig2.savefig('mutual_information_fnn.pdf', format='pdf')
fig2.savefig('mutual_information_fnn.png', format='png', dpi=300)

print("\nMutual Information and FNN analysis saved as:")
print("  - mutual_information_fnn.pdf")
print("  - mutual_information_fnn.png")

# Show all figures
plt.show()

# Print analysis summary
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)
print(f"Signal length: {len(wave)} samples")
print(f"Recommended time delay (tau): {mi_first_min} samples")
print(f"Recommended embedding dimension (d): {optimal_dim}")
print(f"Mutual information at tau={mi_first_min}: {mi_values[mi_first_min - 1]:.4f} bits")
print(f"FNN percentage at d={optimal_dim}: {fnn_values[optimal_dim - 1]:.2f}%")
print("=" * 60)