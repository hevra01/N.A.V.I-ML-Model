import torch
from cost import YoloLoss
from cost import intersection_over_union

img_size = 256 # this value has randomly been assigned
num_classes = 7 # num of classes in kitti dataset


# Create a dummy tensors for testing purposes
# 2 represents the batch size and 3 represents the anchor number
# it is 6 + num_classes because we added distance to model's prediction
# 0th index is objectness, 1st to 4th is bounding box infor, 5th is dist
predictions = torch.randn((2, 3, img_size//32, img_size//32, 6 + num_classes))
target = torch.randn((2, 3, img_size//32, img_size//32, 6 + num_classes))
anchors = torch.tensor([(1, 1), (2, 2), (3, 3)])

# Initialize the YoloLoss module
loss_fn = YoloLoss()

# Compute the loss for the dummy input
loss = loss_fn(predictions, target, anchors)

# Check that loss is finite
assert torch.isfinite(loss), f"Loss has NaN or infinite values."

# Test intersection_over_union function
box1 = torch.tensor([0.5, 0.5, 1.0, 1.0])
box2 = torch.tensor([0.6, 0.6, 1.2, 1.2])
iou = intersection_over_union(box1, box2)
assert iou >= 0 and iou <= 1, "intersection_over_union function is not returning a valid value"

# Verify that YoloLoss module has all the required attributes and methods
assert hasattr(loss_fn, 'mse'), "mse attribute is not defined in the YoloLoss module"
assert hasattr(loss_fn, 'bce'), "bce attribute is not defined in the YoloLoss module"
assert hasattr(loss_fn, 'entropy'), "entropy attribute is not defined in the YoloLoss module"
assert hasattr(loss_fn, 'sigmoid'), "sigmoid attribute is not defined in the YoloLoss module"
assert hasattr(loss_fn, 'lambda_class'), "lambda_class attribute is not defined in the YoloLoss module"
assert hasattr(loss_fn, 'lambda_noobj'), "lambda_noobj attribute is not defined in the YoloLoss module"
assert hasattr(loss_fn, 'lambda_obj'), "lambda_obj attribute is not defined in the YoloLoss module"
assert hasattr(loss_fn, 'lambda_box'), "lambda_box attribute is not defined in the YoloLoss module"
assert hasattr(loss_fn, 'forward'), "forward method is not defined in the YoloLoss module"
assert hasattr(loss_fn, 'lambda_dist'), "lambda_dist attribute is not defined in the YoloLoss module"
