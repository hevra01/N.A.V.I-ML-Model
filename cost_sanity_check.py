import torch
from cost import YoloLoss

# Create a dummy input tensor for testing purposes
predictions = torch.randn((3, 7, 7, 5))
target = torch.randn((3, 7, 7, 6))
anchors = torch.tensor([(1, 1), (2, 2), (3, 3)])

# Initialize the YoloLoss module
loss_fn = YoloLoss()

# Compute the loss for the dummy input
loss = loss_fn(predictions, target, anchors)

# Check that loss is finite
assert torch.isfinite(loss), f"Loss has NaN or infinite values."
