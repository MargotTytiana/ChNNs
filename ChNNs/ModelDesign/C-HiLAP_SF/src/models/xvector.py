import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union


class TDNN(nn.Module):
    """
    Time Delay Neural Network (TDNN) layer for frame-level feature extraction.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        context: List of context offsets to use
        dilation: Dilation factor for convolution

    Reference:
        "X-vectors: Robust DNN Embeddings for Speaker Recognition" (Snyder et al., 2018)
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            context: List[int],
            dilation: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.dilation = dilation

        # Calculate kernel size and padding based on context
        self.kernel_size = max(context) - min(context) + 1

        # Create 1D convolution with specified dilation
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            self.kernel_size,
            dilation=dilation
        )

        # Batch normalization and ReLU activation
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TDNN layer.

        Args:
            x: Input tensor of shape (batch, input_dim, time)

        Returns:
            Output tensor of shape (batch, output_dim, time)
        """
        # Apply convolution
        x = self.conv(x)

        # Apply batch normalization and ReLU
        x = self.bn(x)
        x = self.relu(x)

        return x


class StatisticsPooling(nn.Module):
    """
    Statistics pooling layer that converts frame-level features to segment-level features.
    Computes mean and standard deviation of frame-level features.

    Args:
        dim: Dimension along which to compute statistics (usually time dimension)
    """

    def __init__(self, dim: int = 2):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of statistics pooling layer.

        Args:
            x: Input tensor of shape (batch, feature_dim, time)

        Returns:
            Output tensor of shape (batch, feature_dim * 2)
        """
        # Compute mean along the specified dimension
        mean = torch.mean(x, dim=self.dim)

        # Compute standard deviation along the specified dimension
        std = torch.std(x, dim=self.dim)

        # Concatenate mean and standard deviation
        stats = torch.cat([mean, std], dim=1)

        return stats


class XVector(nn.Module):
    """
    X-vector model for speaker recognition.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of speaker classes (0 for embedding only)
        emb_dim: Embedding dimension

    Reference:
        "X-vectors: Robust DNN Embeddings for Speaker Recognition" (Snyder et al., 2018)
    """

    def __init__(
            self,
            input_dim: int = 40,
            num_classes: int = 0,
            emb_dim: int = 512
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        # Frame-level layers
        self.frame_layers = nn.Sequential(
            # TDNN1: context [-2, -1, 0, 1, 2]
            TDNN(input_dim, 512, context=[-2, -1, 0, 1, 2]),

            # TDNN2: context [-2, 0, 2]
            TDNN(512, 512, context=[-2, 0, 2]),

            # TDNN3: context [-3, 0, 3]
            TDNN(512, 512, context=[-3, 0, 3]),

            # TDNN4: context [0]
            TDNN(512, 512, context=[0]),

            # TDNN5: context [0]
            TDNN(512, 1500, context=[0])
        )

        # Statistics pooling layer
        self.stats_pooling = StatisticsPooling()

        # Segment-level layers
        self.segment_layer1 = nn.Sequential(
            nn.Linear(3000, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )

        # Embedding layer (x-vector)
        self.embedding_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

        # Output layer (optional)
        if num_classes > 0:
            self.output_layer = nn.Linear(emb_dim, num_classes)
        else:
            self.output_layer = None

    def forward(self, x: torch.Tensor, extract_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass of x-vector model.

        Args:
            x: Input tensor of shape (batch, time, input_dim)
            extract_embedding: Whether to return the embedding or the output

        Returns:
            If extract_embedding is True:
                Embedding tensor of shape (batch, emb_dim)
            Else:
                Output tensor of shape (batch, num_classes) if num_classes > 0,
                otherwise embedding tensor of shape (batch, emb_dim)
        """
        # Transpose input to (batch, input_dim, time)
        x = x.transpose(1, 2)

        # Frame-level processing
        x = self.frame_layers(x)

        # Statistics pooling
        x = self.stats_pooling(x)

        # First segment-level layer
        x = self.segment_layer1(x)

        # Embedding layer
        embedding = self.embedding_layer(x)

        # Return embedding if requested
        if extract_embedding:
            return embedding

        # Output layer (if present)
        if self.output_layer is not None:
            output = self.output_layer(embedding)
            return output
        else:
            return embedding

    def extract_xvector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract x-vector embedding from input.

        Args:
            x: Input tensor of shape (batch, time, input_dim)

        Returns:
            X-vector embedding of shape (batch, emb_dim)
        """
        return self.forward(x, extract_embedding=True)


class XVectorClassifier(nn.Module):
    """
    X-vector classifier model that combines x-vector extraction and classification.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of speaker classes
        emb_dim: Embedding dimension
        dropout_prob: Dropout probability for classifier
    """

    def __init__(
            self,
            input_dim: int = 40,
            num_classes: int = 1000,
            emb_dim: int = 512,
            dropout_prob: float = 0.3
    ):
        super().__init__()

        # X-vector extractor
        self.xvector = XVector(input_dim=input_dim, num_classes=0, emb_dim=emb_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, extract_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass of x-vector classifier.

        Args:
            x: Input tensor of shape (batch, time, input_dim)
            extract_embedding: Whether to return the embedding or the output

        Returns:
            If extract_embedding is True:
                Embedding tensor of shape (batch, emb_dim)
            Else:
                Output tensor of shape (batch, num_classes)
        """
        # Extract x-vector
        embedding = self.xvector.extract_xvector(x)

        # Return embedding if requested
        if extract_embedding:
            return embedding

        # Apply classifier
        output = self.classifier(embedding)

        return output


class AMSoftmax(nn.Module):
    """
    Additive Margin Softmax loss for end-to-end speaker verification.

    Args:
        emb_dim: Embedding dimension
        num_classes: Number of speakers
        margin: Margin for AM-Softmax
        scale: Scale factor for AM-Softmax

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
        Forward pass of AM-Softmax loss.

        Args:
            embeddings: Embeddings of shape (batch, emb_dim)
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

        # Add margin to target logits
        phi = cosine - self.margin

        # One-hot encoding for target logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Replace target logits
        output = cosine * (1 - one_hot) + phi * one_hot

        # Apply scale
        output = output * self.scale

        # Calculate loss
        loss = self.criterion(output, labels)

        return loss


class XVectorLossWrapper(nn.Module):
    """
    Wrapper for x-vector model with loss function.

    Args:
        xvector_model: X-vector model
        loss_fn: Loss function
    """

    def __init__(
            self,
            xvector_model: Union[XVector, XVectorClassifier],
            loss_fn: nn.Module
    ):
        super().__init__()
        self.xvector_model = xvector_model
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of x-vector model with loss computation.

        Args:
            x: Input tensor of shape (batch, time, input_dim)
            labels: Ground truth labels of shape (batch,)

        Returns:
            Tuple of (loss, output)
        """
        # Extract embeddings
        embeddings = self.xvector_model(x, extract_embedding=True)

        # Compute output
        output = self.xvector_model(x)

        # Compute loss
        loss = self.loss_fn(embeddings, labels)

        return loss, output


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
    input_dim = 40
    x = torch.randn(batch_size, seq_len, input_dim)

    # Create model
    model = XVector(input_dim=input_dim, num_classes=1000, emb_dim=512)

    # Print model summary
    print(f"Model parameters: {count_parameters(model):,}")

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Extract embeddings
    embeddings = model(x, extract_embedding=True)
    print(f"Embedding shape: {embeddings.shape}")

    # Create classifier model
    classifier = XVectorClassifier(input_dim=input_dim, num_classes=1000)

    # Forward pass
    output = classifier(x)
    print(f"Classifier output shape: {output.shape}")

    # Create AM-Softmax loss
    loss_fn = AMSoftmax(emb_dim=512, num_classes=1000)

    # Create random labels
    labels = torch.randint(0, 1000, (batch_size,))

    # Create loss wrapper
    model_with_loss = XVectorLossWrapper(classifier, loss_fn)

    # Forward pass with loss computation
    loss, output = model_with_loss(x, labels)
    print(f"Loss: {loss.item():.4f}")