import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss # for the bounding box predictions
        self.bce = nn.BCEWithLogitsLoss() # for objectness prediction
        self.entropy = nn.CrossEntropyLoss() # for class prediction
        self.sigmoid = nn.Sigmoid() # activation function

        # Constants/hyperparameters signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        # change: define a loss for distance as well.

