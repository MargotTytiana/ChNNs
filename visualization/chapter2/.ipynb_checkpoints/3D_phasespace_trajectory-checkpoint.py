"""
Multi-Speaker Phase Space Attractor Comparison Visualization.

This script generates a comparison figure showing reconstructed attractors
from different speakers to demonstrate individual specificity of strange
attractors in speech signals.

Author: Margot
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

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


def time_delay_embedding(signal: np.ndarray, delay: int, dimension: int) -> np.ndarray:
    """
    Perform time-delay embedding of a 1D signal.
    
    Args:
        signal: Input 1D time series
        delay: Time delay (tau)
        dimension: Embedding dimension (d_e)
    
    Returns:
        Embedded data matrix of shape (N - (d-1)*tau, d)
    """
    n = len(signal)
    n_embedded = n - (dimension - 1) * delay
    
    if n_embedded <= 0:
        raise ValueError("Signal too short for given delay and dimension")
    
    embedded = np.zeros((n_embedded, dimension))
    for i in range(dimension):
        embedded[:, i] = signal[i * delay : i * delay + n_embedded]
    
    return embedded


def generate_vowel_like_signal(duration: float, fs: int, 
                                f0: float, formants: list,
                                jitter: float = 0.01,
                                shimmer: float = 0.03,
                                noise_level: float = 0.02,
                                seed: int = None) -> np.ndarray:
    """
    Generate a synthetic vowel-like signal with chaotic characteristics.
    
    Args:
        duration: Signal duration in seconds
        fs: Sampling frequency
        f0: Fundamental frequency (pitch)
        formants: List of formant frequencies [F1, F2, F3]
        jitter: Pitch perturbation factor
        shimmer: Amplitude perturbation factor  
        noise_level: Additive noise level
        seed: Random seed for reproducibility
    
    Returns:
        Synthetic vowel signal
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # Generate glottal pulse train with jitter
    glottal = np.zeros(n_samples)
    period = fs / f0
    current_pos = 0
    
    while current_pos < n_samples:
        # Add jitter to period
        jittered_period = period * (1 + jitter * np.random.randn())
        jittered_period = max(period * 0.8, min(period * 1.2, jittered_period))
        
        # Glottal pulse (simplified LF model)
        pulse_len = int(jittered_period * 0.4)
        if current_pos + pulse_len < n_samples:
            pulse_t = np.linspace(0, np.pi, pulse_len)
            pulse = np.sin(pulse_t) ** 2
            # Add shimmer
            pulse *= (1 + shimmer * np.random.randn())
            glottal[int(current_pos):int(current_pos) + pulse_len] = pulse
        
        current_pos += jittered_period
    
    # Apply formant filtering (resonances)
    signal = glottal.copy()
    for formant in formants:
        # Create resonance effect
        bandwidth = formant * 0.1
        resonance = np.exp(-bandwidth * t / fs) * np.sin(2 * np.pi * formant * t)
        signal = np.convolve(signal, resonance[:min(500, len(resonance))], mode='same')
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    
    # Add turbulent noise (aspiration)
    noise = noise_level * np.random.randn(n_samples)
    signal += noise
    
    # Add subtle nonlinear distortion (characteristic of vocal fold dynamics)
    signal = np.tanh(2 * signal)
    
    return signal


def generate_speaker_characteristics():
    """
    Generate characteristic parameters for different simulated speakers.
    
    Returns:
        Dictionary of speaker parameters
    """
    speakers = {
        'Speaker A': {
            'f0': 120,  # Male speaker, lower pitch
            'formants': [700, 1200, 2600],  # Vowel /a/
            'jitter': 0.008,
            'shimmer': 0.025,
            'noise_level': 0.015,
            'color': '#2E86AB',
            'description': 'Male, vowel /a/'
        },
        'Speaker B': {
            'f0': 220,  # Female speaker, higher pitch
            'formants': [400, 2200, 2800],  # Vowel /i/
            'jitter': 0.012,
            'shimmer': 0.035,
            'noise_level': 0.02,
            'color': '#E94F37',
            'description': 'Female, vowel /i/'
        },
        'Speaker C': {
            'f0': 150,  # Another male speaker
            'formants': [500, 1800, 2500],  # Vowel /e/
            'jitter': 0.015,
            'shimmer': 0.04,
            'noise_level': 0.025,
            'color': '#44AF69',
            'description': 'Male, vowel /e/'
        }
    }
    return speakers


def plot_attractor_comparison():
    """
    Create the main comparison figure showing attractors from different speakers.
    """
    # Parameters
    duration = 0.3  # 300ms vowel segment
    fs = 16000  # 16kHz sampling rate
    delay = 8  # Time delay for embedding
    dimension = 3  # 3D embedding
    
    # Get speaker characteristics
    speakers = generate_speaker_characteristics()
    
    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.1, wspace=0.2)
    
    # Title for the entire figure
    fig.suptitle('Reconstructed Phase Space Attractors: Speaker Comparison', 
                 fontsize=14, fontweight='bold', y=1)
    
    # Top row: Same speaker (Speaker A), different recording sessions
    # Bottom row: Different speakers
    
    # ===== TOP ROW: Same Speaker (Speaker A), Different Sessions =====
    speaker_a_params = speakers['Speaker A']
    sessions = [
        {'seed': 42, 'label': 'Session 1'},
        {'seed': 123, 'label': 'Session 2'},
        {'seed': 456, 'label': 'Session 3'}
    ]
    
    for col, session in enumerate(sessions):
        # Generate signal for this session
        signal = generate_vowel_like_signal(
            duration=duration,
            fs=fs,
            f0=speaker_a_params['f0'],
            formants=speaker_a_params['formants'],
            jitter=speaker_a_params['jitter'],
            shimmer=speaker_a_params['shimmer'],
            noise_level=speaker_a_params['noise_level'],
            seed=session['seed']
        )
        
        # Perform time-delay embedding
        embedded = time_delay_embedding(signal, delay, dimension)
        
        # Create 3D subplot
        ax = fig.add_subplot(gs[0, col], projection='3d')
        
        # Plot trajectory
        ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                color=speaker_a_params['color'], alpha=0.7, linewidth=0.5)
        
        # Mark starting point
        ax.scatter(embedded[0, 0], embedded[0, 1], embedded[0, 2],
                   color='red', s=30, zorder=5, marker='o')
        
        # Set labels and title
        ax.set_xlabel('$s(t)$', fontsize=9, labelpad=-12)
        ax.set_ylabel('$s(t+\\tau)$', fontsize=9, labelpad=-12)
        ax.set_zlabel('$s(t+2\\tau)$', fontsize=9, labelpad=-12)
        ax.text2D(0.5, 0.9, f"Speaker A - {session['label']}", 
                  fontsize=11, ha='center', transform=ax.transAxes)
        
        # Adjust view angle for consistency
        ax.view_init(elev=20, azim=45)
        
        # Remove tick labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    # Add row label
    fig.text(0.08, 0.73, 'Same Speaker\n(Different Sessions)', 
             fontsize=11, fontweight='bold', va='center', rotation=90,
             color='#333333')
    
    # ===== BOTTOM ROW: Different Speakers =====
    speaker_list = ['Speaker A', 'Speaker B', 'Speaker C']
    
    for col, speaker_name in enumerate(speaker_list):
        params = speakers[speaker_name]
        
        # Generate signal
        signal = generate_vowel_like_signal(
            duration=duration,
            fs=fs,
            f0=params['f0'],
            formants=params['formants'],
            jitter=params['jitter'],
            shimmer=params['shimmer'],
            noise_level=params['noise_level'],
            seed=42  # Same seed for fair comparison
        )
        
        # Perform time-delay embedding
        embedded = time_delay_embedding(signal, delay, dimension)
        
        # Create 3D subplot
        ax = fig.add_subplot(gs[1, col], projection='3d')
        
        # Plot trajectory
        ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                color=params['color'], alpha=0.7, linewidth=0.5)
        
        # Mark starting point
        ax.scatter(embedded[0, 0], embedded[0, 1], embedded[0, 2],
                   color='red', s=30, zorder=5, marker='o')
        
        # Set labels and title
        ax.set_xlabel('$s(t)$', fontsize=9, labelpad=-12)
        ax.set_ylabel('$s(t+\\tau)$', fontsize=9, labelpad=-12)
        ax.set_zlabel('$s(t+2\\tau)$', fontsize=9, labelpad=-12)
        ax.text2D(0.5, 0.9, f"{speaker_name}\n({params['description']})", 
                  fontsize=11, ha='center', transform=ax.transAxes)
        
        # Adjust view angle
        ax.view_init(elev=20, azim=45)
        
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    # Add row label
    fig.text(0.08, 0.3, 'Different Speakers\n(Same Conditions)', 
             fontsize=11, fontweight='bold', va='center', rotation=90,
             color='#333333')
    
    # Add explanatory text box
    textstr = '\n'.join([
        f'$\\cdot$ Session: Independent recording of the',
        f'  same vowel by the same speaker at different times',
        f'$\\cdot$ Red dot: Trajectory starting point',
        f'$\\cdot$ Embedding parameters: $d_e$={dimension}, $\\tau$={delay}',
        f'$\\cdot$ Signal: 300ms vowel, fs=16kHz'
    ])
    
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', 
                 edgecolor='#cccccc', alpha=0.9)
    fig.text(0.7, 0.01, textstr, fontsize=9, verticalalignment='bottom',
             horizontalalignment='left', bbox=props, family='Helvetica')
    
    plt.subplots_adjust(top=0.94, hspace=0.12, wspace=0.10)
    
    # Save figure
    plt.savefig('figure_output/speaker_attractor_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure_output/speaker_attractor_comparison.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("Figure saved successfully!")
    print("  - PNG: figure_output/speaker_attractor_comparison.png")
    print("  - PDF: figure_output/speaker_attractor_comparison.pdf")
    
    plt.show()


def plot_2d_projection_comparison():
    """
    Create an additional 2D projection comparison for clearer visualization.
    """
    # Parameters
    duration = 0.3
    fs = 16000
    delay = 8
    dimension = 3
    
    speakers = generate_speaker_characteristics()
    
    # Create figure with more left margin for row labels
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle('2D Phase Space Projections: Speaker Comparison', 
                 fontsize=14, fontweight='bold', y=0.94)
    
    # Top row: Same speaker, different sessions
    speaker_a_params = speakers['Speaker A']
    sessions = [{'seed': 42}, {'seed': 123}, {'seed': 456}]
    
    for col, session in enumerate(sessions):
        signal = generate_vowel_like_signal(
            duration=duration, fs=fs,
            f0=speaker_a_params['f0'],
            formants=speaker_a_params['formants'],
            jitter=speaker_a_params['jitter'],
            shimmer=speaker_a_params['shimmer'],
            noise_level=speaker_a_params['noise_level'],
            seed=session['seed']
        )
        embedded = time_delay_embedding(signal, delay, dimension)
        
        ax = axes[0, col]
        ax.plot(embedded[:, 0], embedded[:, 1], 
                color=speaker_a_params['color'], alpha=0.6, linewidth=0.3)
        ax.scatter(embedded[0, 0], embedded[0, 1], color='red', s=20, zorder=5)
        ax.set_xlabel('$s(t)$')
        ax.set_ylabel('$s(t+\\tau)$')
        ax.set_title(f'Speaker A - Session {col+1}')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
    
    # Bottom row: Different speakers
    for col, (speaker_name, params) in enumerate(speakers.items()):
        signal = generate_vowel_like_signal(
            duration=duration, fs=fs,
            f0=params['f0'],
            formants=params['formants'],
            jitter=params['jitter'],
            shimmer=params['shimmer'],
            noise_level=params['noise_level'],
            seed=42
        )
        embedded = time_delay_embedding(signal, delay, dimension)
        
        ax = axes[1, col]
        ax.plot(embedded[:, 0], embedded[:, 1], 
                color=params['color'], alpha=0.6, linewidth=0.3)
        ax.scatter(embedded[0, 0], embedded[0, 1], color='red', s=20, zorder=5)
        ax.set_xlabel('$s(t)$')
        ax.set_ylabel('$s(t+\\tau)$')
        ax.set_title(f"{speaker_name}\n({params['description']})")
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
    
    # Adjust layout first to get proper spacing
    plt.tight_layout(rect=[0.08, 0, 1, 0.95])
    
    # Add row labels using figure coordinates (outside the axes area)
    fig.text(0.08, 0.72, 'Same Speaker', fontsize=11, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.08, 0.28, 'Different Speakers', fontsize=11, fontweight='bold', 
             rotation=90, va='center', ha='center')
    
    plt.savefig('figure_output/speaker_attractor_2d_projection.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure_output/speaker_attractor_2d_projection.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("2D projection figure saved!")
    plt.show()


if __name__ == "__main__":
    print("Generating Multi-Speaker Phase Space Attractor Comparison...")
    print("=" * 60)
    
    # Generate main 3D comparison figure
    plot_attractor_comparison()
    
    # Generate 2D projection figure
    plot_2d_projection_comparison()
    
    print("=" * 60)
    print("All figures generated successfully!")