import torch
import torch.nn as nn
import utils
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss # for the bounding box predictions, and distance estimation
        self.bce = nn.BCEWithLogitsLoss() # for objectness prediction
        self.entropy = nn.CrossEntropyLoss() # for class prediction
        self.sigmoid = nn.Sigmoid() # activation function

        # Constants/hyperparameters signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        self.lambda_dist = 10  # for the distance prediction

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        print("predictions: ", predictions)

        print("target: ", target)
        print(predictions.shape)
        print(target.shape)

        # [..., 0] => gets the zeroth index of all the rows, which is about whether an object is present or not
        # 1 means object is present, -1 means the object is not present, hence there is no need to
        # punish the model for incorrect class or distance estimation since there is no object anyways
        obj = target[..., 0] == 1
        noobj = target[..., 0] == -1

        print(obj)
        print(noobj)
        print(predictions[..., 0:1][noobj])

        # ======================= #
        #   FOR NO OBJECT LOSS
        #   This is to penalize anchors that didn't predict the existence of an object
        #   by only the objectness loss and not misclassification or bounding box loss
        # ======================= #
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS
        #   This is for anchors that predicted that an object exists.
        #   The loss will include IOU (intersection over union)#
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors],
                              dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = torch.mean((self.sigmoid(predictions[..., 0:1][obj]) - (ious * target[..., 0:1][obj]))**2)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = torch.mean((predictions[..., 1:5][obj] - target[..., 1:5][obj])**2)

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.entropy(
            # indexing all the values from the 6th channel to the end of the tensor.
            # because until the 6th channel it contains info about: 1 for objectness,
            # bounding box (4 values), and 1 for distance. indexing starts from 0.
            (predictions[..., 6:][obj]), (target[..., 6][obj].long()),
        )

        # ==================== #
        #   FOR DISTANCE LOSS   #
        # ==================== #

        dist_targets = target[..., 5][obj]
        dist_predictions = predictions[..., 5][obj]

        dist_loss = self.mse(
            dist_predictions, dist_targets,
        )

        print("__________________________________")
        print(self.lambda_box * box_loss)
        print(self.lambda_obj * object_loss)
        print(self.lambda_noobj * no_object_loss)
        print(self.lambda_class * class_loss)
        print(self.lambda_dist * dist_loss)
        print("\n")

        return (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
                + self.lambda_dist * dist_loss
        )