import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation (SE) module for channel-wise attention.

    Args:
        channels: Number of input channels
        bottleneck: Bottleneck dimension for SE block

    Reference:
        "Squeeze-and-Excitation Networks" (Hu et al., 2018)
    """

    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SE module.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Tensor of shape (batch, channels, time) with channel attention applied
        """
        return x * self.se(x)


class Res2Block(nn.Module):
    """
    Res2Net block with Squeeze-and-Excitation.

    Args:
        channels: Number of input and output channels
        scale: Scale factor for Res2Net
        dilation: Dilation factor for convolutions
        kernel_size: Kernel size for convolutions

    Reference:
        "Res2Net: A New Multi-scale Backbone Architecture" (Gao et al., 2019)
    """

    def __init__(self, channels: int, scale: int = 8, dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        self.scale = scale
        self.channels = channels
        self.width = channels // scale

        # Convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.width,
                self.width,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2
            ) for _ in range(scale - 1)
        ])

        # Batch normalization layers
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width) for _ in range(scale - 1)
        ])

        # Squeeze-and-Excitation module
        self.se = SEModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Res2Block.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Output tensor of shape (batch, channels, time)
        """
        residual = x

        # Split input along channel dimension
        xs = torch.split(x, self.width, dim=1)
        ys = []

        # First part goes through as identity
        ys.append(xs[0])

        # Process remaining parts with convolutions
        for i in range(self.scale - 1):
            if i == 0:
                ys.append(self.bns[i](F.relu(self.convs[i](xs[i + 1]))))
            else:
                ys.append(self.bns[i](F.relu(self.convs[i](ys[-1] + xs[i + 1]))))

        # Concatenate outputs
        y = torch.cat(ys, dim=1)

        # Apply SE module
        y = self.se(y)

        # Add residual connection
        y = y + residual

        return y


class SERes2NetBlock(nn.Module):
    """
    SE-Res2Net block as used in ECAPA-TDNN.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for convolutions
        dilation: Dilation factor for convolutions
        scale: Scale factor for Res2Net
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            dilation: int = 1,
            scale: int = 8
    ):
        super().__init__()

        # First 1x1 convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Res2Block
        self.res2block = Res2Block(
            out_channels,
            scale=scale,
            dilation=dilation,
            kernel_size=kernel_size
        )

        # Second 1x1 convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SE-Res2Net block.

        Args:
            x: Input tensor of shape (batch, in_channels, time)

        Returns:
            Output tensor of shape (batch, out_channels, time)
        """
        # First convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Res2Block
        x = self.res2block(x)

        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x)

        return x


class AttentiveStatsPool(nn.Module):
    """
    Attentive Statistics Pooling layer for aggregating frame-level features.

    Args:
        in_dim: Input feature dimension
        bottleneck_dim: Bottleneck dimension for attention

    Reference:
        "Attentive Statistics Pooling for Deep Speaker Embedding" (Okabe et al., 2018)
    """

    def __init__(self, in_dim: int, bottleneck_dim: int = 128):
        super().__init__()

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )

        # Output dimension is twice the input dimension (mean + std)
        self.output_dim = in_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attentive statistics pooling.

        Args:
            x: Input tensor of shape (batch, in_dim, time)

        Returns:
            Output tensor of shape (batch, in_dim * 2)
        """
        # Calculate attention weights
        attention_weights = self.attention(x)

        # Apply attention weights
        mean = torch.sum(x * attention_weights, dim=2)

        # Calculate variance with attention weights
        var = torch.sum(x ** 2 * attention_weights, dim=2) - mean ** 2

        # Handle numerical issues
        std = torch.sqrt(torch.clamp(var, min=1e-9))

        # Concatenate mean and standard deviation
        pooled = torch.cat([mean, std], dim=1)

        return pooled


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN model for speaker recognition.

    Args:
        input_dim: Input feature dimension
        channels: Number of channels in convolutional blocks
        emb_dim: Final embedding dimension

    Reference:
        "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
        TDNN Based Speaker Verification" (Desplanques et al., 2020)
    """

    def __init__(
            self,
            input_dim: int = 80,
            channels: int = 512,
            emb_dim: int = 192
    ):
        super().__init__()

        # Initial 1x1 convolution
        self.conv1 = nn.Conv1d(input_dim, channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        # Frame-level blocks with different dilations
        self.layer1 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=1, scale=8)
        self.layer2 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=2, scale=8)
        self.layer3 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=3, scale=8)
        self.layer4 = SERes2NetBlock(channels, channels, kernel_size=3, dilation=4, scale=8)

        # Multilayer feature aggregation
        self.mfa = nn.Conv1d(channels * 4, channels, kernel_size=1)
        self.bn_mfa = nn.BatchNorm1d(channels)

        # Attentive statistics pooling
        self.asp = AttentiveStatsPool(channels)

        # Final embedding layers
        self.fc1 = nn.Linear(channels * 2, channels)
        self.bn_fc1 = nn.BatchNorm1d(channels)
        self.fc2 = nn.Linear(channels, emb_dim)
        self.bn_fc2 = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor, extract_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass of ECAPA-TDNN model.

        Args:
            x: Input tensor of shape (batch, time, input_dim)
            extract_embedding: Whether to return the embedding or the normalized embedding

        Returns:
            If extract_embedding is False:
                Normalized embedding of shape (batch, emb_dim)
            Else:
                Embedding of shape (batch, emb_dim)
        """
        # Transpose input to (batch, input_dim, time)
        x = x.transpose(1, 2)

        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Frame-level feature extraction
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer1_out + layer2_out)
        layer4_out = self.layer4(layer1_out + layer2_out + layer3_out)

        # Multi-layer feature aggregation
        x = torch.cat([layer1_out, layer2_out, layer3_out, layer4_out], dim=1)
        x = self.mfa(x)
        x = self.bn_mfa(x)
        x = self.relu(x)

        # Attentive statistics pooling
        x = self.asp(x)

        # Final embedding layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        if extract_embedding:
            return x

        # Apply batch normalization for normalized embeddings
        x = self.bn_fc2(x)

        return x


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax loss for end-to-end speaker verification.

    Args:
        emb_dim: Embedding dimension
        num_classes: Number of speakers
        margin: Margin for AAM-Softmax
        scale: Scale factor for AAM-Softmax

    Reference:
        "Additive Margin Softmax for Face Verification" (Wang et al., 2018)
    """

    def __init__(
            self,
            emb_dim: int,
            num_classes: int,
            margin: float = 0.2,
            scale: float = 30.0
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale

        # Weight for each class
        self.weight = nn.Parameter(torch.randn(num_classes, emb_dim))
        self.weight.data.normal_(0, 0.01)  # Initialize with small random values

        # Criterion for computing loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of AAM-Softmax loss.

        Args:
            embeddings: Normalized embeddings of shape (batch, emb_dim)
            labels: Ground truth labels of shape (batch,)

        Returns:
            Loss value
        """
        # Normalize weights
        normalized_weight = F.normalize(self.weight, dim=1)

        # Normalize embeddings
        normalized_embeddings = F.normalize(embeddings, dim=1)

        # Calculate cosine similarity
        cosine = F.linear(normalized_embeddings, normalized_weight)

        # Get target logits
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        # Add margin to target logits
        target_logits = torch.cos(theta + self.margin)

        # One-hot encoding for target logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Replace target logits
        output = cosine * (1 - one_hot) + target_logits * one_hot

        # Apply scale
        output = output * self.scale

        # Calculate loss
        loss = self.criterion(output, labels)

        return loss


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    batch_size = 2
    seq_len = 200
    input_dim = 80
    x = torch.randn(batch_size, seq_len, input_dim)

    # Create model
    model = ECAPA_TDNN(input_dim=input_dim, channels=512, emb_dim=192)

    # Print model summary
    print(f"Model parameters: {count_parameters(model):,}")

    # Forward pass
    embeddings = model(x)
    print(f"Output shape: {embeddings.shape}")

    # Extract embeddings
    raw_embeddings = model(x, extract_embedding=True)
    print(f"Raw embedding shape: {raw_embeddings.shape}")

    # Create AAM-Softmax loss
    num_speakers = 10
    loss_fn = AAMSoftmax(emb_dim=192, num_classes=num_speakers)

    # Create random labels
    labels = torch.randint(0, num_speakers, (batch_size,))

    # Calculate loss
    loss = loss_fn(raw_embeddings, labels)
    print(f"Loss: {loss.item():.4f}")
