import torch
import torch.nn as nn

import config
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()  # for the bounding box predictions, and distance estimation
        self.bce = nn.BCEWithLogitsLoss()  # for objectness prediction
        self.entropy = nn.CrossEntropyLoss()  # for class prediction
        self.sigmoid = nn.Sigmoid()  # activation function

        # Constants/hyperparameters signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        self.lambda_dist = 10  # for the distance prediction

    def forward(self, predictions, target, anchors):
        # [..., 0] => gets the zeroth index of all the rows, which is about whether an object is present or not
        # 1 means object is present, 0 means the object is not present, hence there is no need to
        # punish the model for incorrect class or distance estimation since there is no object anyways
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # ======================= #
        #   FOR NO OBJECT LOSS
        #   This is to penalize anchors that didn't predict the existence of an object
        #   by only the objectness loss and not misclassification or bounding box loss
        # ======================= #

        # find entropy of the class predictions returned by the class.
        # if the entropy is high, the noobj lose needs to be close to zero.
        # if the entropy is low, the noonj lose needs to be high.
        # the 11th index of the model output is objectness score
        model_noobj_prediction = predictions[..., 11][noobj].clone()
        no_object_loss = self.bce(
            model_noobj_prediction, (target[..., 0][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS
        #   This is for anchors that predicted that an object exists.
        #   The loss will include IOU (intersection over union)#
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        bb_xy_to_be_sigmoided = predictions[..., 7:9].clone()
        bb_wh_to_be_sigmoided = predictions[..., 9:11].clone()
        box_preds = torch.cat([self.sigmoid(bb_xy_to_be_sigmoided), torch.exp(bb_wh_to_be_sigmoided) * anchors],
                              dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        to_be_unsqueezed = predictions[..., 11][obj].clone()
        model_obj_prediction = to_be_unsqueezed.unsqueeze(1)
        object_loss = self.bce(model_obj_prediction, (ious * target[..., 0:1][obj]))

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        target_cloned = target[..., 3:5].clone()
        # target[..., 3:5] = torch.log(1e-16 + target_cloned / anchors)
        target = torch.cat([target[..., :3], torch.log(1e-16 + target_cloned / anchors), target[..., 5:]], dim=-1)

        # apply sigmoid to x, y coordinates to convert to bounding boxes
        sigmoid_input = predictions[..., 7:9].clone()
        # predictions[..., 7:9] = self.sigmoid(sigmoid_input.clone())
        predictions = torch.cat([predictions[..., :7], self.sigmoid(sigmoid_input), predictions[..., 9:]], dim=-1)

        bb_predicted_by_model = predictions[..., 7:11][obj].clone()
        # compute mse loss for boxes
        box_loss = torch.sqrt(self.mse(bb_predicted_by_model, target[..., 1:5][obj]))  # mean squared logarithmic error

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_confidences_predicted_by_model = predictions[..., :7][obj].clone()
        class_loss = self.entropy(
            # indexing all the values from the 6th channel to the end of the tensor.
            # because until the 6th channel it contains info about: 1 for objectness,
            # bounding box (4 values), and 1 for distance. indexing starts from 0.
            class_confidences_predicted_by_model, (target[..., 5][obj].long()),
        )

        # ==================== #
        #   FOR DISTANCE LOSS   #
        # ==================== #

        # Make sure both tensors have the same shape
        assert target[..., 6][obj].shape == predictions[..., -1][obj].shape

        dist_targets = target[..., 6][obj].clone()
        # the model's last prediction is distance hence -1 to get the last element
        dist_predictions = predictions[..., -1][obj].clone()
        correct_dist = (abs(dist_predictions - dist_targets)) <= config.CONF_DIST_THRESHOLD
        correct_dist = torch.sum(correct_dist)
        dist_loss = 1 - (correct_dist.item() / dist_targets.shape[0])
        return (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
                + self.lambda_dist * dist_loss
        )
