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

        def forward(self, predictions, target, anchors):
            # Check where obj and noobj (we ignore if target == -1)
            obj = target[..., 0] == 1  # in paper this is Iobj_i
            noobj = target[..., 0] == 0  # in paper this is Inoobj_i

            # ======================= #
            #   FOR NO OBJECT LOSS
            #   This is to penalize anchors that didn't predict the existence of an object
            #   by only the objectness loss and not misclassification or bounding box loss
            # ======================= #

            no_object_loss = self.bce(
                (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
            )

            # ==================== #
            #   FOR OBJECT LOSS    #
            # ==================== #

            anchors = anchors.reshape(1, 3, 1, 1, 2)
            box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors],
                                  dim=-1)
            ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
            object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])




