"""ModernCNN composition and fit-mode forward-pass reference.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html
"""

import brainstate
import brainunit as u


class ModernCNN(brainstate.nn.Module):
    """CNN with modern activations and normalization."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1: Conv + BatchNorm + GELU
        self.conv1 = brainstate.nn.Conv2d((32, 32, 3), out_channels=64, kernel_size=(3, 3), padding='SAME')
        self.bn1 = brainstate.nn.BatchNorm2d((32, 32, 64))
        self.act1 = brainstate.nn.GELU()
        self.pool1 = brainstate.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 2
        self.conv2 = brainstate.nn.Conv2d((16, 16, 64), out_channels=128, kernel_size=(3, 3), padding='SAME')
        self.bn2 = brainstate.nn.BatchNorm2d((16, 16, 128))
        self.act2 = brainstate.nn.GELU()
        self.pool2 = brainstate.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), in_size=self.bn2.out_size)

        # Classifier
        self.fc1 = brainstate.nn.Linear((128 * 8 * 8,), (256,))
        self.ln = brainstate.nn.LayerNorm((256,))
        self.act3 = brainstate.nn.GELU()
        self.dropout = brainstate.nn.Dropout(prob=0.5)
        self.fc2 = brainstate.nn.Linear((256,), (num_classes,))

    def update(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        # Classifier
        x = u.math.flatten(x, start_axis=1)
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Create and test
brainstate.random.seed(0)
model = ModernCNN(num_classes=10)

# Forward pass
x = brainstate.random.randn(4, 32, 32, 3)  # 4 images
with brainstate.environ.context(fit=True) as env:
    logits = model(x)

print("Modern CNN with GELU + BatchNorm + LayerNorm:")
print(model)
print()
print("Input:", x.shape)
print("Output:", logits.shape)
print()
print("Logits:", logits[0])
