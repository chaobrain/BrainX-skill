"""Residual connections and dynamic child registration for BrainState Modules.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/02_modules_and_graph.html
Role: Mirrors the tutorial's custom helpers and ``ResidualBlock``/``ResNet``
example as a standalone script.
"""

import brainstate
import jax.numpy as jnp


class Linear(brainstate.nn.Module):
    """A linear transformation: y = W @ x + b"""

    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # Initialize weight with Xavier/Glorot initialization
        std = jnp.sqrt(2.0 / (in_features + out_features))
        self.weight = brainstate.ParamState(
            brainstate.random.randn(in_features, out_features) * std
        )

        # Initialize bias to zero
        if use_bias:
            self.bias = brainstate.ParamState(jnp.zeros(out_features))

    def update(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        out = x @ self.weight.value
        if self.use_bias:
            out = out + self.bias.value
        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, use_bias={self.use_bias})"


class LeakyReLU(brainstate.nn.Module):
    """Leaky ReLU activation: y = max(alpha * x, x)"""

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def update(self, x):
        return jnp.where(x > 0, x, self.negative_slope * x)

    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class ResidualBlock(brainstate.nn.Module):
    """Residual block: y = F(x) + x"""

    def __init__(self, dim):
        super().__init__()

        # Two linear layers with activation in between
        self.linear1 = Linear(dim, dim)
        self.activation = LeakyReLU(0.0)
        self.linear2 = Linear(dim, dim)

    def update(self, x):
        # Compute residual
        residual = x

        # Forward through layers
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)

        # Add residual
        return out + residual


class ResNet(brainstate.nn.Module):
    """Simple ResNet with multiple residual blocks."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks=3):
        super().__init__()

        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = []
        for i in range(n_blocks):
            block = ResidualBlock(hidden_dim)
            setattr(self, f"block_{i}", block)
            self.blocks.append(block)

        # Output projection
        self.output_proj = Linear(hidden_dim, output_dim)

    def update(self, x):
        # Project to hidden dimension
        x = self.input_proj(x)

        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)

        # Project to output
        x = self.output_proj(x)
        return x


# Create ResNet
brainstate.random.seed(0)
resnet = ResNet(input_dim=10, hidden_dim=32, output_dim=5, n_blocks=3)

# Forward pass
x = brainstate.random.randn(10)
y = resnet(x)

print("ResNet:")
print(resnet)
print(f"\nOutput shape: {y.shape}")
