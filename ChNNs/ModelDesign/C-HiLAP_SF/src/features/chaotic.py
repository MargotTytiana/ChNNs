import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings


class ChaoticFeatureExtractor:
    """
    Extracts chaos-based features from speech signals including:
    - Maximum Lyapunov Exponent (MLE)
    - Recurrence Quantification Analysis (RQA) metrics
    - Phase space reconstruction parameters
    - Fractal dimensions
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 frame_length: int = 25,  # ms
                 frame_shift: int = 10,  # ms
                 embedding_dim: int = 10,
                 delay: int = 5,
                 radius: float = 0.1,
                 theiler_window: int = 10):
        """
        Initialize chaotic feature extractor with configuration parameters.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_length: Frame length in milliseconds
            frame_shift: Frame shift in milliseconds
            embedding_dim: Embedding dimension for phase space reconstruction
            delay: Time delay for embedding
            radius: Threshold radius for recurrence plot
            theiler_window: Theiler window to exclude temporal correlations
        """
        self.sample_rate = sample_rate
        self.frame_length = int(frame_length * sample_rate / 1000)
        self.frame_shift = int(frame_shift * sample_rate / 1000)
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.radius = radius
        self.theiler_window = theiler_window

    def preemphasize(self, signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Apply preemphasis filter to enhance high frequencies.

        Args:
            signal: Input audio signal
            coef: Preemphasis coefficient

        Returns:
            Preemphasized signal
        """
        return np.append(signal[0], signal[1:] - coef * signal[:-1])

    def framing(self,
                signal: np.ndarray,
                frame_length: Optional[int] = None,
                frame_shift: Optional[int] = None) -> np.ndarray:
        """
        Split signal into overlapping frames.

        Args:
            signal: Input audio signal
            frame_length: Frame length in samples (optional)
            frame_shift: Frame shift in samples (optional)

        Returns:
            Framed signal of shape (num_frames, frame_length)
        """
        if frame_length is None:
            frame_length = self.frame_length
        if frame_shift is None:
            frame_shift = self.frame_shift

        signal_length = len(signal)
        num_frames = 1 + int(np.floor((signal_length - frame_length) / frame_shift))

        frames = np.zeros((num_frames, frame_length))
        for i in range(num_frames):
            frames[i] = signal[i * frame_shift: i * frame_shift + frame_length]

        return frames

    def reconstruct_phase_space(self, signal: np.ndarray) -> np.ndarray:
        """
        Reconstruct phase space using time delay embedding (Takens' theorem).

        Args:
            signal: Input signal (single frame)

        Returns:
            Phase space trajectory of shape (n_points, embedding_dim)
        """
        n = len(signal)
        m = self.embedding_dim
        tau = self.delay

        # Number of points in the reconstructed space
        n_points = n - (m - 1) * tau

        if n_points <= 0:
            # If signal is too short, reduce embedding dimension or delay
            reduced_m = min(m, n // 2)
            reduced_tau = 1
            n_points = n - (reduced_m - 1) * reduced_tau

            warnings.warn(
                f"Signal too short for embedding. Reducing dimension to {reduced_m} "
                f"and delay to {reduced_tau}."
            )

            m = reduced_m
            tau = reduced_tau

        # Initialize the reconstructed phase space
        phase_space = np.zeros((n_points, m))

        # Fill the phase space
        for i in range(n_points):
            for j in range(m):
                phase_space[i, j] = signal[i + j * tau]

        return phase_space

    def compute_recurrence_plot(self,
                                phase_space: np.ndarray,
                                radius: Optional[float] = None,
                                norm_type: str = 'euclidean') -> np.ndarray:
        """
        Compute recurrence plot from phase space trajectory.

        Args:
            phase_space: Phase space trajectory of shape (n_points, embedding_dim)
            radius: Threshold radius (optional)
            norm_type: Type of norm ('euclidean', 'manhattan', 'max')

        Returns:
            Recurrence plot matrix of shape (n_points, n_points)
        """
        if radius is None:
            radius = self.radius

        n_points = phase_space.shape[0]
        rp = np.zeros((n_points, n_points), dtype=np.uint8)

        # Compute distances and thresholding
        for i in range(n_points):
            for j in range(i + self.theiler_window, n_points):
                if norm_type == 'euclidean':
                    dist = np.sqrt(np.sum((phase_space[i] - phase_space[j]) ** 2))
                elif norm_type == 'manhattan':
                    dist = np.sum(np.abs(phase_space[i] - phase_space[j]))
                elif norm_type == 'max':
                    dist = np.max(np.abs(phase_space[i] - phase_space[j]))
                else:
                    raise ValueError(f"Unknown norm type: {norm_type}")

                if dist <= radius:
                    rp[i, j] = rp[j, i] = 1

        return rp

    def compute_rqa_metrics(self, rp: np.ndarray) -> Dict[str, float]:
        """
        Compute Recurrence Quantification Analysis metrics from recurrence plot.

        Args:
            rp: Recurrence plot matrix

        Returns:
            Dictionary of RQA metrics
        """
        n_points = rp.shape[0]

        # Recurrence Rate (RR)
        rr = np.sum(rp) / (n_points ** 2)

        # Find diagonal lines (simplified approach)
        min_diag_length = 2
        diag_lengths = []

        for k in range(-(n_points - min_diag_length), n_points - min_diag_length + 1):
            diag = np.diag(rp, k=k)
            if len(diag) >= min_diag_length:
                # Find consecutive ones
                consec_ones = np.diff(np.hstack(([0], diag, [0])))
                # Start indices of consecutive ones
                starts = np.where(consec_ones == 1)[0]
                # End indices of consecutive ones
                ends = np.where(consec_ones == -1)[0]
                # Lengths of diagonal lines
                lengths = ends - starts

                for length in lengths:
                    if length >= min_diag_length:
                        diag_lengths.append(length)

        # Determinism (DET)
        if len(diag_lengths) > 0 and np.sum(rp) > 0:
            det = np.sum(diag_lengths) / np.sum(rp)
        else:
            det = 0.0

        # Average Diagonal Line Length (L)
        if len(diag_lengths) > 0:
            avg_diag_length = np.mean(diag_lengths)
        else:
            avg_diag_length = 0.0

        # Maximum Diagonal Line Length (Lmax)
        if len(diag_lengths) > 0:
            max_diag_length = np.max(diag_lengths)
        else:
            max_diag_length = 0.0

        # Divergence (DIV) - inverse of Lmax
        div = 1.0 / (max_diag_length + 1e-10)

        # Find vertical lines (simplified approach)
        min_vert_length = 2
        vert_lengths = []

        for j in range(n_points):
            col = rp[:, j]
            # Find consecutive ones
            consec_ones = np.diff(np.hstack(([0], col, [0])))
            # Start indices of consecutive ones
            starts = np.where(consec_ones == 1)[0]
            # End indices of consecutive ones
            ends = np.where(consec_ones == -1)[0]
            # Lengths of vertical lines
            lengths = ends - starts

            for length in lengths:
                if length >= min_vert_length:
                    vert_lengths.append(length)

        # Laminarity (LAM)
        if len(vert_lengths) > 0 and np.sum(rp) > 0:
            lam = np.sum(vert_lengths) / np.sum(rp)
        else:
            lam = 0.0

        # Trapping Time (TT)
        if len(vert_lengths) > 0:
            tt = np.mean(vert_lengths)
        else:
            tt = 0.0

        # Entropy measures
        if len(diag_lengths) > 0:
            # Compute histogram of diagonal line lengths
            hist, _ = np.histogram(diag_lengths, bins=range(2, max(diag_lengths) + 2))
            prob = hist / np.sum(hist)
            # Shannon entropy
            entropy = -np.sum(prob * np.log(prob + 1e-10))
        else:
            entropy = 0.0

        return {
            'recurrence_rate': float(rr),
            'determinism': float(det),
            'avg_diag_length': float(avg_diag_length),
            'max_diag_length': float(max_diag_length),
            'divergence': float(div),
            'laminarity': float(lam),
            'trapping_time': float(tt),
            'entropy': float(entropy)
        }

    def compute_mle(self,
                    signal: np.ndarray,
                    min_neighbors: int = 5,
                    max_iterations: int = 20) -> float:
        """
        Compute Maximum Lyapunov Exponent using the algorithm of Rosenstein et al.

        Args:
            signal: Input signal (single frame)
            min_neighbors: Minimum number of neighbors to consider
            max_iterations: Maximum number of iterations

        Returns:
            Maximum Lyapunov exponent
        """
        # Phase space reconstruction
        phase_space = self.reconstruct_phase_space(signal)

        if len(phase_space) < min_neighbors + 1:
            return 0.0

        n_points = phase_space.shape[0]
        m = self.embedding_dim

        # Find nearest neighbors (excluding temporal neighbors)
        mean_divergence = np.zeros(max_iterations)
        neighbor_count = 0

        for i in range(n_points):
            # Compute distances to all other points
            distances = np.zeros(n_points)
            for j in range(n_points):
                if abs(i - j) <= self.theiler_window:
                    distances[j] = float('inf')  # Exclude temporal neighbors
                else:
                    distances[j] = np.sqrt(np.sum((phase_space[i] - phase_space[j]) ** 2))

            # Find nearest neighbor
            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] == float('inf'):
                continue

            # Track divergence
            for k in range(max_iterations):
                if i + k < n_points and nearest_idx + k < n_points:
                    d_k = np.sqrt(np.sum((phase_space[i + k] - phase_space[nearest_idx + k]) ** 2))
                    if d_k > 0:
                        mean_divergence[k] += np.log(d_k)

            neighbor_count += 1

        if neighbor_count == 0:
            return 0.0

        # Average divergence
        mean_divergence = mean_divergence / neighbor_count

        # Linear regression to find the slope (MLE)
        x = np.arange(max_iterations)
        valid_idx = ~np.isnan(mean_divergence) & ~np.isinf(mean_divergence)

        if np.sum(valid_idx) < 2:
            return 0.0

        x = x[valid_idx]
        y = mean_divergence[valid_idx]

        # Linear fit
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

        # Convert to time units
        mle = slope / (self.frame_shift / self.sample_rate)

        return float(mle)

    def compute_correlation_dimension(self,
                                      signal: np.ndarray,
                                      max_radius: float = 0.5,
                                      n_points: int = 20) -> float:
        """
        Compute correlation dimension (D2) using the Grassberger-Procaccia algorithm.

        Args:
            signal: Input signal (single frame)
            max_radius: Maximum radius for correlation sum
            n_points: Number of radius values to use

        Returns:
            Correlation dimension (D2)
        """
        # Phase space reconstruction
        phase_space = self.reconstruct_phase_space(signal)

        if len(phase_space) < 10:  # Need enough points
            return 0.0

        n_samples = phase_space.shape[0]

        # Generate logarithmically spaced radii
        radii = np.logspace(-3, np.log10(max_radius), n_points)

        # Compute correlation sum for each radius
        corr_sum = np.zeros(len(radii))

        for i, r in enumerate(radii):
            count = 0
            for j in range(n_samples):
                for k in range(j + self.theiler_window, n_samples):
                    dist = np.sqrt(np.sum((phase_space[j] - phase_space[k]) ** 2))
                    if dist < r:
                        count += 1

            # Normalize
            if n_samples > self.theiler_window:
                pairs = (n_samples * (n_samples - self.theiler_window - 1)) // 2
                if pairs > 0:
                    corr_sum[i] = count / pairs

        # Avoid log(0)
        valid_idx = corr_sum > 0
        if np.sum(valid_idx) < 2:
            return 0.0

        log_radii = np.log(radii[valid_idx])
        log_corr_sum = np.log(corr_sum[valid_idx])

        # Linear fit to find the slope (D2)
        A = np.vstack([log_radii, np.ones(len(log_radii))]).T
        slope, _ = np.linalg.lstsq(A, log_corr_sum, rcond=None)[0]

        return float(slope)

    def compute_sample_entropy(self,
                               signal: np.ndarray,
                               m: int = 2,
                               r: float = 0.2) -> float:
        """
        Compute sample entropy of the signal.

        Args:
            signal: Input signal (single frame)
            m: Embedding dimension
            r: Tolerance (typically 0.1-0.25 * std(signal))

        Returns:
            Sample entropy value
        """
        # Normalize signal
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

        n = len(signal)

        # Tolerance
        r = r * np.std(signal)

        # Initialize count arrays
        count_m = 0
        count_m_plus_1 = 0

        # Create templates of length m and m+1
        for i in range(n - m):
            template_m = signal[i:i + m]
            template_m_plus_1 = signal[i:i + m + 1]

            # Count matches for template_m
            for j in range(i + 1, n - m + 1):
                if np.max(np.abs(template_m - signal[j:j + m])) < r:
                    count_m += 1

                    # Check if m+1 also matches
                    if j <= n - m - 1:
                        if np.max(np.abs(template_m_plus_1 - signal[j:j + m + 1])) < r:
                            count_m_plus_1 += 1

        # Compute sample entropy
        if count_m == 0 or count_m_plus_1 == 0:
            return 0.0
        else:
            return -np.log(count_m_plus_1 / count_m)

    def extract_chaotic_features(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all chaotic features from audio signal.

        Args:
            signal: Input audio signal

        Returns:
            Dictionary containing all extracted chaotic features
        """
        # Apply preemphasis
        preemphasized = self.preemphasize(signal)

        # Frame the signal
        frames = self.framing(preemphasized)

        # Initialize feature arrays
        num_frames = frames.shape[0]
        mle_features = np.zeros(num_frames)
        cd_features = np.zeros(num_frames)
        se_features = np.zeros(num_frames)
        rqa_features = np.zeros((num_frames, 8))  # 8 RQA metrics

        # Extract features for each frame
        for i in range(num_frames):
            frame = frames[i]

            # Compute Maximum Lyapunov Exponent
            mle_features[i] = self.compute_mle(frame)

            # Compute Correlation Dimension
            cd_features[i] = self.compute_correlation_dimension(frame)

            # Compute Sample Entropy
            se_features[i] = self.compute_sample_entropy(frame)

            # Compute RQA metrics
            phase_space = self.reconstruct_phase_space(frame)
            rp = self.compute_recurrence_plot(phase_space)
            rqa_metrics = self.compute_rqa_metrics(rp)

            rqa_features[i, 0] = rqa_metrics['recurrence_rate']
            rqa_features[i, 1] = rqa_metrics['determinism']
            rqa_features[i, 2] = rqa_metrics['avg_diag_length']
            rqa_features[i, 3] = rqa_metrics['max_diag_length']
            rqa_features[i, 4] = rqa_metrics['divergence']
            rqa_features[i, 5] = rqa_metrics['laminarity']
            rqa_features[i, 6] = rqa_metrics['trapping_time']
            rqa_features[i, 7] = rqa_metrics['entropy']

        return {
            'mle': mle_features,
            'correlation_dimension': cd_features,
            'sample_entropy': se_features,
            'rqa': rqa_features
        }

    def extract_recurrence_plot_image(self,
                                      signal: np.ndarray,
                                      downsample_factor: int = 1) -> np.ndarray:
        """
        Extract recurrence plot image from audio signal.

        Args:
            signal: Input audio signal
            downsample_factor: Factor to downsample the signal by

        Returns:
            Recurrence plot image
        """
        # Apply preemphasis
        preemphasized = self.preemphasize(signal)

        # Downsample if needed
        if downsample_factor > 1:
            preemphasized = preemphasized[::downsample_factor]

        # Phase space reconstruction
        phase_space = self.reconstruct_phase_space(preemphasized)

        # Compute recurrence plot
        rp = self.compute_recurrence_plot(phase_space)

        return rp

    def visualize_recurrence_plot(self,
                                  rp: np.ndarray,
                                  title: str = "Recurrence Plot") -> plt.Figure:
        """
        Visualize recurrence plot.

        Args:
            rp: Recurrence plot matrix
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rp, cmap='binary', origin='lower')
        ax.set_title(title)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Time Index")

        return fig

    def visualize_phase_space(self,
                              signal: np.ndarray,
                              title: str = "Phase Space Reconstruction") -> plt.Figure:
        """
        Visualize phase space reconstruction.

        Args:
            signal: Input signal
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Phase space reconstruction
        phase_space = self.reconstruct_phase_space(signal)

        if phase_space.shape[1] < 3:
            # If embedding dimension is less than 3, pad with zeros
            padded = np.zeros((phase_space.shape[0], 3))
            padded[:, :phase_space.shape[1]] = phase_space
            phase_space = padded

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the first three dimensions
        ax.plot(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2], 'b-', lw=0.5)
        ax.set_title(title)
        ax.set_xlabel("X(t)")
        ax.set_ylabel("X(t+τ)")
        ax.set_zlabel("X(t+2τ)")

        return fig


class ChaoticOscillator(nn.Module):
    """
    Base class for chaotic oscillators that can be used in neural networks.
    Implements various chaotic systems as ODEs that can be solved with torchdiffeq.
    """

    def __init__(self, system_type: str = "lorenz"):
        """
        Initialize chaotic oscillator.

        Args:
            system_type: Type of chaotic system ('lorenz', 'rossler', 'chen', 'chua')
        """
        super().__init__()
        self.system_type = system_type

        # Initialize parameters for different systems
        if system_type == "lorenz":
            # Lorenz system parameters
            self.sigma = nn.Parameter(torch.tensor(10.0))
            self.rho = nn.Parameter(torch.tensor(28.0))
            self.beta = nn.Parameter(torch.tensor(8.0 / 3.0))
        elif system_type == "rossler":
            # Rössler system parameters
            self.a = nn.Parameter(torch.tensor(0.2))
            self.b = nn.Parameter(torch.tensor(0.2))
            self.c = nn.Parameter(torch.tensor(5.7))
        elif system_type == "chen":
            # Chen system parameters
            self.a = nn.Parameter(torch.tensor(35.0))
            self.b = nn.Parameter(torch.tensor(3.0))
            self.c = nn.Parameter(torch.tensor(28.0))
        elif system_type == "chua":
            # Chua's circuit parameters
            self.alpha = nn.Parameter(torch.tensor(15.6))
            self.beta = nn.Parameter(torch.tensor(28.0))
            self.m0 = nn.Parameter(torch.tensor(-1.143))
            self.m1 = nn.Parameter(torch.tensor(-0.714))
        else:
            raise ValueError(f"Unknown system type: {system_type}")

    def forward(self, t, state):
        """
        Compute the derivative of the state for the chaotic system.

        Args:
            t: Time (not used in autonomous systems)
            state: Current state [x, y, z]

        Returns:
            State derivative [dx/dt, dy/dt, dz/dt]
        """
        if self.system_type == "lorenz":
            x, y, z = state
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            return torch.stack([dx, dy, dz])

        elif self.system_type == "rossler":
            x, y, z = state
            dx = -y - z
            dy = x + self.a * y
            dz = self.b + z * (x - self.c)
            return torch.stack([dx, dy, dz])

        elif self.system_type == "chen":
            x, y, z = state
            dx = self.a * (y - x)
            dy = (self.c - self.a) * x - x * z + self.c * y
            dz = x * y - self.b * z
            return torch.stack([dx, dy, dz])

        elif self.system_type == "chua":
            x, y, z = state
            # Chua's diode nonlinearity
            h_x = self.m1 * x + 0.5 * (self.m0 - self.m1) * (torch.abs(x + 1) - torch.abs(x - 1))

            dx = self.alpha * (y - h_x)
            dy = x - y + z
            dz = -self.beta * y
            return torch.stack([dx, dy, dz])


class ControlledChaoticOscillator(nn.Module):
    """
    Chaotic oscillator with external control input from neural network.
    Can be used as a layer in neural networks to introduce chaotic dynamics.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 system_type: str = "lorenz",
                 integration_time: float = 1.0,
                 integration_steps: int = 10,
                 learn_parameters: bool = True):
        """
        Initialize controlled chaotic oscillator.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state
            system_type: Type of chaotic system
            integration_time: Time to integrate the ODE
            integration_steps: Number of integration steps
            learn_parameters: Whether to learn system parameters
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.system_type = system_type
        self.integration_time = integration_time
        self.integration_steps = integration_steps

        # Input projection
        self.input_proj = nn.Linear(input_dim, 3)

        # Initialize chaotic system
        self.oscillator = ChaoticOscillator(system_type)

        # Control matrix (coupling strength)
        self.control_matrix = nn.Parameter(torch.randn(3, 3) * 0.01)

        # Output projection
        self.output_proj = nn.Linear(3, hidden_dim)

        # Freeze system parameters if not learning
        if not learn_parameters:
            for param in self.oscillator.parameters():
                param.requires_grad = False

    def system_with_control(self, t, state, control_input):
        """
        Chaotic system with external control input.

        Args:
            t: Time
            state: Current state
            control_input: External control input

        Returns:
            State derivative
        """
        # Get base system dynamics
        dstate = self.oscillator(t, state)

        # Add control term
        control_term = torch.matmul(self.control_matrix, control_input)

        return dstate + control_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the controlled chaotic oscillator.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to control signal
        control_input = self.input_proj(x)  # (batch_size, seq_len, 3)

        # Initialize output tensor
        output = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)

        # Process each sequence step
        for b in range(batch_size):
            # Initialize state with first control input
            state = control_input[b, 0]

            for t in range(seq_len):
                # Current control input
                control = control_input[b, t]

                # Integration time points
                t_span = torch.linspace(0, self.integration_time, self.integration_steps, device=x.device)

                # Define ODE function with current control input
                def controlled_system(t, y):
                    return self.system_with_control(t, y, control)

                # Solve ODE
                trajectory = odeint(
                    controlled_system,
                    state,
                    t_span,
                    method='rk4',
                    options=dict(step_size=self.integration_time / self.integration_steps)
                )

                # Update state with last point of trajectory
                state = trajectory[-1]

                # Project to output space
                output[b, t] = self.output_proj(state)

        return output


class ChaoticAttentionLayer(nn.Module):
    """
    Attention mechanism controlled by chaotic dynamics.
    Uses bifurcation parameters to control attention behavior.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 bifurcation_factor: float = 1.0):
        """
        Initialize chaotic attention layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bifurcation_factor: Factor controlling chaotic behavior
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.bifurcation_factor = bifurcation_factor

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Bifurcation parameter (learnable)
        self.r = nn.Parameter(torch.tensor(3.5))  # Initial value in chaotic regime

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def logistic_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply logistic map iteration to introduce chaos.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        # Normalize to [0, 1] range for logistic map
        x_norm = torch.sigmoid(x)

        # Apply logistic map with bifurcation parameter
        r_scaled = self.bifurcation_factor * torch.sigmoid(self.r) * 4.0  # Scale to [0, 4]
        return r_scaled * x_norm * (1 - x_norm)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of chaotic attention layer.

        Args:
            query: Query tensor of shape (batch_size, tgt_len, embed_dim)
            key: Key tensor of shape (batch_size, src_len, embed_dim)
            value: Value tensor of shape (batch_size, src_len, embed_dim)
            key_padding_mask: Mask for padding in key
            attn_mask: Mask for attention weights

        Returns:
            Output tensor of shape (batch_size, tgt_len, embed_dim)
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Apply chaotic transformation
        q = self.logistic_map(q)
        k = self.logistic_map(k)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply masks if provided
        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)

        # Output projection
        output = self.out_proj(output)

        return output


class StrangeAttractorPooling(nn.Module):
    """
    Pooling layer based on strange attractor dynamics.
    Uses chaotic system to integrate temporal information.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 system_type: str = "lorenz",
                 integration_steps: int = 20):
        """
        Initialize strange attractor pooling.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            system_type: Type of chaotic system
            integration_steps: Number of integration steps
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.system_type = system_type
        self.integration_steps = integration_steps

        # Input projection
        self.input_proj = nn.Linear(input_dim, 3)

        # Chaotic oscillator
        self.oscillator = ChaoticOscillator(system_type)

        # Output projection
        self.output_proj = nn.Linear(3, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of strange attractor pooling.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to initial conditions
        init_conditions = self.input_proj(x[:, 0])  # Use first frame as initial condition

        # Initialize states for each batch
        states = init_conditions

        # Integrate through time using the sequence
        for t in range(seq_len):
            # Current input
            x_t = self.input_proj(x[:, t])

            # Define time span for integration
            t_span = torch.linspace(0, 1.0, self.integration_steps, device=x.device)

            # Integrate for each batch item
            for b in range(batch_size):
                # Define ODE with current input as parameter
                def system_with_input(t, state):
                    # Base dynamics
                    dstate = self.oscillator(t, state)

                    # Add input influence
                    input_influence = 0.1 * (x_t[b] - state)

                    return dstate + input_influence

                # Solve ODE
                trajectory = odeint(
                    system_with_input,
                    states[b],
                    t_span,
                    method='rk4'
                )

                # Update state
                states[b] = trajectory[-1]

        # Project final states to output dimension
        output = self.output_proj(states)

        return output


class ChaoticEmbeddingLayer(nn.Module):
    """
    Embedding layer with chaotic dynamics for speaker recognition.
    Transforms input features using chaotic maps to enhance discriminability.
    """

    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 chaos_type: str = "tent",
                 ks_entropy: float = 0.5):
        """
        Initialize chaotic embedding layer.

        Args:
            input_dim: Input feature dimension
            embed_dim: Embedding dimension
            chaos_type: Type of chaotic map ('logistic', 'tent', 'henon')
            ks_entropy: Target Kolmogorov-Sinai entropy
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.chaos_type = chaos_type
        self.ks_entropy = ks_entropy

        # Linear projection
        self.proj = nn.Linear(input_dim, embed_dim)

        # Chaotic parameters (learnable)
        if chaos_type == "logistic":
            # Logistic map: x_{n+1} = r * x_n * (1 - x_n)
            self.r = nn.Parameter(torch.tensor(3.9))  # Bifurcation parameter
        elif chaos_type == "tent":
            # Tent map: x_{n+1} = mu * min(x_n, 1-x_n)
            self.mu = nn.Parameter(torch.tensor(1.9))  # Slope parameter
        elif chaos_type == "henon":
            # Henon map: x_{n+1} = 1 - a*x_n^2 + b*y_n, y_{n+1} = x_n
            self.a = nn.Parameter(torch.tensor(1.4))
            self.b = nn.Parameter(torch.tensor(0.3))
        else:
            raise ValueError(f"Unknown chaos type: {chaos_type}")

        # Mixing matrix
        self.mixing = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)

    def apply_chaos(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply chaotic map to input tensor.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        # Normalize to appropriate range for the map
        x_norm = torch.sigmoid(x)  # [0, 1]

        if self.chaos_type == "logistic":
            # Logistic map
            r = torch.sigmoid(self.r) * 4.0  # Scale to [0, 4]
            return r * x_norm * (1 - x_norm)

        elif self.chaos_type == "tent":
            # Tent map
            mu = torch.sigmoid(self.mu) * 2.0  # Scale to [0, 2]
            return mu * torch.min(x_norm, 1 - x_norm)

        elif self.chaos_type == "henon":
            # Henon map (simplified 1D version)
            a = torch.sigmoid(self.a) * 2.0  # Scale to [0, 2]
            b = torch.sigmoid(self.b)  # Scale to [0, 1]

            # Apply map and rescale to [0, 1]
            x_out = 1 - a * x_norm ** 2 + b * (1 - x_norm)
            return torch.sigmoid(x_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of chaotic embedding layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Embedded tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Linear projection
        x = self.proj(x)

        # Apply chaotic transformation
        x = self.apply_chaos(x)

        # Apply mixing
        x = torch.matmul(x, self.mixing)

        return x


class CHiLAP(nn.Module):
    """
    Chaotic Hierarchical Attractor Propagation (C-HiLAP) model for speaker recognition.
    Complete implementation of the proposed architecture.
    """

    def __init__(self,
                 input_dim: int = 80,
                 embed_dim: int = 512,
                 num_classes: int = 0,
                 num_layers: int = 4,
                 chaos_type: str = "lorenz",
                 ks_entropy: float = 0.5,
                 bifurcation_factor: float = 1.0):
        """
        Initialize C-HiLAP model.

        Args:
            input_dim: Input feature dimension
            embed_dim: Embedding dimension
            num_classes: Number of speaker classes (0 for embedding only)
            num_layers: Number of model layers
            chaos_type: Type of chaotic system
            ks_entropy: Target Kolmogorov-Sinai entropy
            bifurcation_factor: Factor controlling chaotic behavior
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Input embedding
        self.input_embed = ChaoticEmbeddingLayer(
            input_dim=input_dim,
            embed_dim=embed_dim,
            chaos_type="tent",
            ks_entropy=ks_entropy
        )

        # Chaotic layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                # Controlled chaotic oscillator
                'chaos': ControlledChaoticOscillator(
                    input_dim=embed_dim,
                    hidden_dim=embed_dim,
                    system_type=chaos_type,
                    integration_time=1.0,
                    integration_steps=10
                ),
                # Chaotic attention
                'attention': ChaoticAttentionLayer(
                    embed_dim=embed_dim,
                    num_heads=8,
                    dropout=0.1,
                    bifurcation_factor=bifurcation_factor
                ),
                # Feed-forward network
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, 4 * embed_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(4 * embed_dim, embed_dim)
                ),
                # Layer norms
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim)
            })
            self.layers.append(layer)

        # Strange attractor pooling
        self.pooling = StrangeAttractorPooling(
            input_dim=embed_dim,
            output_dim=embed_dim,
            system_type=chaos_type
        )

        # Output projection
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor, extract_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass of C-HiLAP model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            extract_embedding: Whether to return embeddings only

        Returns:
            Output tensor of shape (batch_size, num_classes) or (batch_size, embed_dim)
        """
        # Input embedding
        x = self.input_embed(x)

        # Process through layers
        for layer in self.layers:
            # Chaotic processing
            chaos_out = layer['chaos'](x)
            x = layer['norm1'](x + chaos_out)

            # Self-attention
            attn_out = layer['attention'](x, x, x)
            x = x + attn_out

            # Feed-forward
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)

        # Strange attractor pooling
        embedding = self.pooling(x)

        # Return embedding if requested
        if extract_embedding:
            return embedding

        # Classification
        if self.classifier is not None:
            return self.classifier(embedding)
        else:
            return embedding


class LyapunovRegularization(nn.Module):
    """
    Regularization based on Lyapunov stability theory.
    Encourages stable or chaotic dynamics based on the task.
    """

    def __init__(self,
                 target_exponent: float = 0.1,
                 weight: float = 0.1,
                 mode: str = "positive"):
        """
        Initialize Lyapunov regularization.

        Args:
            target_exponent: Target Lyapunov exponent
            weight: Weight of regularization term
            mode: 'positive' for chaotic, 'negative' for stable, 'target' for specific value
        """
        super().__init__()
        self.target_exponent = target_exponent
        self.weight = weight
        self.mode = mode

    def estimate_lyapunov(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Estimate largest Lyapunov exponent from trajectory.

        Args:
            trajectory: Tensor of shape (batch_size, seq_len, dim)

        Returns:
            Tensor of shape (batch_size,) containing Lyapunov exponents
        """
        batch_size, seq_len, dim = trajectory.shape

        # Need sufficient length for estimation
        if seq_len < 10:
            return torch.zeros(batch_size, device=trajectory.device)

        # Compute divergence of nearby trajectories
        # (simplified implementation)
        divergence = torch.zeros(batch_size, device=trajectory.device)

        for b in range(batch_size):
            # Use first half as reference
            ref_points = trajectory[b, :seq_len // 2]

            # Compute distances between consecutive time steps
            diffs = torch.norm(ref_points[1:] - ref_points[:-1], dim=1)

            # Estimate exponential divergence rate
            if torch.any(diffs > 0):
                log_diffs = torch.log(diffs + 1e-10)
                divergence[b] = torch.mean(log_diffs)

        return divergence

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute Lyapunov regularization loss.

        Args:
            trajectory: Tensor of shape (batch_size, seq_len, dim)

        Returns:
            Regularization loss
        """
        # Estimate Lyapunov exponents
        lyapunov = self.estimate_lyapunov(trajectory)

        if self.mode == "positive":
            # Encourage chaotic dynamics (positive exponents)
            loss = torch.mean(torch.relu(-lyapunov))
        elif self.mode == "negative":
            # Encourage stable dynamics (negative exponents)
            loss = torch.mean(torch.relu(lyapunov))
        else:  # "target"
            # Encourage specific exponent value
            loss = torch.mean((lyapunov - self.target_exponent) ** 2)

        return self.weight * loss


class PhaseSynchronizationLoss(nn.Module):
    """
    Loss function based on phase synchronization between input and attractor.
    Encourages the model to synchronize with specific phase patterns.
    """

    def __init__(self, weight: float = 0.1):
        """
        Initialize phase synchronization loss.

        Args:
            weight: Weight of loss term
        """
        super().__init__()
        self.weight = weight

    def extract_phase(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Extract instantaneous phase using analytic signal approach.

        Args:
            signal: Input signal of shape (batch_size, seq_len, dim)

        Returns:
            Phase tensor of shape (batch_size, seq_len, dim)
        """
        # Implementation uses a simplified approach since Hilbert transform
        # is not directly available in PyTorch

        batch_size, seq_len, dim = signal.shape
        phases = torch.zeros_like(signal)

        for b in range(batch_size):
            for d in range(dim):
                # Convert to numpy for processing
                sig = signal[b, :, d].cpu().detach().numpy()

                # Compute analytic signal (simplified)
                sig_mean = np.mean(sig)
                sig_centered = sig - sig_mean

                # Find zero crossings as phase reference points
                zero_crossings = np.where(np.diff(np.signbit(sig_centered)))[0]

                if len(zero_crossings) >= 2:
                    # Estimate local frequency
                    avg_period = (zero_crossings[-1] - zero_crossings[0]) / (len(zero_crossings) - 1)

                    # Generate phase based on zero crossings
                    phase = np.zeros(seq_len)
                    last_crossing = 0
                    phase_val = 0

                    for zc in zero_crossings:
                        phase[last_crossing:zc + 1] = np.linspace(
                            phase_val, phase_val + np.pi, zc - last_crossing + 1
                        )
                        last_crossing = zc + 1
                        phase_val += np.pi

                    # Fill remaining values
                    if last_crossing < seq_len:
                        remaining = seq_len - last_crossing
                        phase[last_crossing:] = np.linspace(
                            phase_val, phase_val + np.pi * remaining / avg_period, remaining
                        )

                    # Wrap to [-π, π]
                    phase = (phase + np.pi) % (2 * np.pi) - np.pi

                    # Convert back to tensor
                    phases[b, :, d] = torch.tensor(phase, device=signal.device)

        return phases

    def forward(self, input_signal: torch.Tensor, attractor_signal: torch.Tensor) -> torch.Tensor:
        """
        Compute phase synchronization loss.

        Args:
            input_signal: Input signal
            attractor_signal: Attractor signal

        Returns:
            Synchronization loss
        """
        # Extract phases
        input_phase = self.extract_phase(input_signal)
        attractor_phase = self.extract_phase(attractor_signal)

        # Compute phase difference
        phase_diff = input_phase - attractor_phase

        # Compute synchronization index (1 - R)
        # where R is the phase locking value
        sync_loss = 1.0 - torch.abs(torch.mean(torch.exp(1j * phase_diff.float())))

        return self.weight * sync_loss


# Example usage
if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt

    # Load a sample audio file
    audio_file = "P:/PycharmProjects/pythonProject1/dataset/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
    signal, sr = librosa.load(audio_file, sr=16000)

    # Create chaotic feature extractor
    feature_extractor = ChaoticFeatureExtractor(sample_rate=sr)

    # Extract chaotic features
    chaotic_features = feature_extractor.extract_chaotic_features(signal)

    # Extract recurrence plot
    rp = feature_extractor.extract_recurrence_plot_image(signal, downsample_factor=10)

    # Visualize recurrence plot
    fig = feature_extractor.visualize_recurrence_plot(rp, title="Recurrence Plot of Speech Signal")
    fig.savefig("recurrence_plot.png")

    # Visualize phase space
    fig = feature_extractor.visualize_phase_space(signal[:1000], title="Phase Space Reconstruction")
    fig.savefig("phase_space.png")

    # Plot MLE and RQA features
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title("Maximum Lyapunov Exponent")
    plt.plot(chaotic_features['mle'])
    plt.xlabel("Frame")
    plt.ylabel("MLE")

    plt.subplot(2, 2, 2)
    plt.title("Correlation Dimension")
    plt.plot(chaotic_features['correlation_dimension'])
    plt.xlabel("Frame")
    plt.ylabel("D2")

    plt.subplot(2, 2, 3)
    plt.title("Sample Entropy")
    plt.plot(chaotic_features['sample_entropy'])
    plt.xlabel("Frame")
    plt.ylabel("SampEn")

    plt.subplot(2, 2, 4)
    plt.title("RQA: Determinism")
    plt.plot(chaotic_features['rqa'][:, 1])
    plt.xlabel("Frame")
    plt.ylabel("DET")

    plt.tight_layout()
    plt.savefig("chaotic_features.png")
    print("Saved visualizations to disk")

    # Create a small test for the C-HiLAP model
    batch_size = 2
    seq_len = 50
    input_dim = 80

    # Create random input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Initialize model
    model = CHiLAP(
        input_dim=input_dim,
        embed_dim=512,
        num_classes=10,
        num_layers=2,
        chaos_type="lorenz",
        ks_entropy=0.5,
        bifurcation_factor=1.0
    )

    # Forward pass
    output = model(x)
    embeddings = model(x, extract_embedding=True)

    print(f"Output shape: {output.shape}")
    print(f"Embedding shape: {embeddings.shape}")
