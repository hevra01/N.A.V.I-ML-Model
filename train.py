import itertools
import pickle

import cv2
import numpy as np
import torch
import torch.optim as optim
from scipy import stats
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

import config
from cost import YoloLoss
from model import YOLOv3
from utils import (
    check_class_accuracy,
    get_loaders
)

torch.backends.cudnn.benchmark = True

import torch.nn as nn


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)


# this function will help us find the average performance of our model using cross-validation

def cross_validation(model, loss_fn, scaler, whole_dataset, scaled_anchors, hyperparameters):
    # Define the number of folds for cross-validation
    k_folds = KFold(n_splits=10, shuffle=True, random_state=42)

    # initialize to zero.
    avg_class_accuracy = 0
    avg_obj_accuracy = 0
    avg_no_obj_accuracy = 0
    avg_distance_accuracy = 0
    print("before for loop")
    # iterate over k-folds.
    # enumerate(k_folds.split()) returns the fold index (fold) and
    # a tuple ((train_indices, val_indices)) containing the indices of the training and
    # validation sets for the current fold.
    for fold, (train_indices, val_indices) in enumerate(k_folds.split(range(whole_dataset.__len__()))):
        # Create the train and validation subsets using Subset
        train_subset = Subset(whole_dataset, train_indices)
        val_subset = Subset(whole_dataset, val_indices)

        # for debugging purposes. yes, nice! 90% goes for training, 10% goes for testing
        t_percent = len(train_subset) / whole_dataset.__len__()
        v_percent = len(val_subset) / whole_dataset.__len__()
        print("train_subset percentage: ", t_percent, "val_subset percentage: ", v_percent)
        # Create the train and validation loaders using DataLoader
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            drop_last=False,
        )

        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )
        print("after creating train and eval loaders!")
        # perform training on k-1 folds
        optimizer = optim.Adam(
            model.parameters(), lr=hyperparameters[2], weight_decay=config.WEIGHT_DECAY
        )
        for epoch in range(config.NUM_EPOCHS):
            train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        print("after train")

        # perform testing on the kth fold
        class_accuracy, obj_accuracy, no_obj_accuracy, distance_accuracy = check_class_accuracy(model, val_loader,
                                                                                                threshold=
                                                                                                hyperparameters[0],
                                                                                                dist_threshold=
                                                                                                hyperparameters[1])

        print("after check_accuracy")
        # here, we are accumulating the accuracy, then we will divide by the number of folds
        avg_class_accuracy += class_accuracy
        avg_obj_accuracy += obj_accuracy
        avg_no_obj_accuracy += no_obj_accuracy
        avg_distance_accuracy += distance_accuracy

        if config.SAVE_MODEL:
            name = "fold_" + str(fold) + "_model.pk"
            with open(name, 'wb') as file:
                pickle.dump(model, file)
                file.close()

    # in order to get the average, we need to divide by the number of folds
    avg_class_accuracy /= k_folds
    avg_obj_accuracy /= k_folds
    avg_no_obj_accuracy /= k_folds
    avg_distance_accuracy /= k_folds

    return avg_class_accuracy, avg_obj_accuracy, avg_no_obj_accuracy, avg_distance_accuracy


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    # creates a progress bar for the train_loader iterable object,
    # which is typically a DataLoader object used in PyTorch for loading training data in batches.
    loop = tqdm(train_loader, leave=True)
    losses = []

    # enumerate() function to iterate over each batch of data in the train_loader object.
    # batch_idx contains the index number of the current batch,
    # while (x, y) contains the inputs x and labels y for the current batch.
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        # performing forward pass and loss computation in a mixed precision setting using PyTorch's AMP feature.
        with torch.cuda.amp.autocast():
            out = model(x)
            # change: this is kinda weird because in the dataset I don't think there are more than one label per image
            # not for multiple scales I mean, maybe we are creating them in the dataloader. check this please
            loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
            )

        # adds the current batch loss to a losses list
        losses.append(loss.item())
        # The optimizer.zero_grad() function is used to set the gradients of all
        # the model parameters to zero before computing the gradients of the current
        # batch in the training loop. This is important because PyTorch accumulates
        # gradients on subsequent backward passes, so if you don't zero out the gradients
        # before each batch, the gradients will accumulate across batches and lead to
        # incorrect updates of the model parameters.
        optimizer.zero_grad()
        # backward pass and computes gradients with respect to the loss.
        scaler.scale(loss).backward()
        # updates the model parameters by taking an optimizer step.
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        median_loss = stats.trim_mean(losses, 0.1)
        # sets the progress bar's display to show the current mean loss value
        # the loss on the progress bar is being updated after every batch
        loop.set_postfix(loss=median_loss)


# this function gets the output of the model, which is predictions in three
# different scales, and combines them


def predict(model, img):
    bbox = []
    class_confidences = []
    class_ids = []
    distances = []

    # The network predicts offsets (x, y, w, h) for each anchor box, where (x, y) represents the center coordinates
    # of the bounding box and (w, h) represents its width and height. Additionally, class probabilities are
    # predicted for each anchor box to determine the object class it belongs to.
    predictions = model(img)

    # prediction in 3 scales
    for scale in predictions:
        # iterate over the batch. we usually send only one image at a time
        for image in scale:
            # there are 3 anchors
            for anchor in image:
                # iterate over the bounding boxes horizontally
                for gridx in anchor:
                    # iterate over the bounding boxes vertically
                    for gridy in gridx:
                        # get the class predictions
                        scores = gridy[:7]
                        distance = gridy[-1]
                        object_confidence = gridy[11]  # objectnessScore
                        class_id = np.argmax(scores.cpu().detach().numpy())
                        class_confidence = scores[class_id]
                        # we need to add objectness to the model prediction as well.
                        # Discard bad detections and continue.
                        if object_confidence > config.OBJ_PRESENCE_CONFIDENCE_THRESHOLD:
                            center_x = int(gridy[7] * config.IMAGE_SIZE)
                            center_y = int(gridy[8] * config.IMAGE_SIZE)
                            w = int(gridy[9] * config.IMAGE_SIZE)
                            h = int(gridy[9] * config.IMAGE_SIZE)

                            x = int(center_x - (w / 2))
                            y = int(center_y - (h / 2))

                            bbox.append([x, y, w, h])
                            class_ids.append(class_id)
                            class_confidences.append(max(scores.cpu().detach().numpy()))
                            # last element of the prediction is the distance
                            distances.append(distance)

    # Now we have got all the bounding boxes, but we need only one bounding box for each object,
    # so we pass the bounding boxes to NMS function.
    # Based on the confidence threshold and nms threshold the boxes will get suppressed, and we will
    # get the box which detects the object correctly.

    detected_object = []
    print(class_confidences)
    indices = cv2.dnn.NMSBoxes(bbox, class_confidences, config.CLASS_CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)
    print(indices)
    for i in indices:
        x, y, w, h = bbox[i]
        label = config.KITTI_CLASSES[class_ids[i]]
        distance = distances[i]
        confidence = class_confidences[i]
        detected_object.append([x, y, w, h, label, confidence, distance.item()])

    print(detected_object)
    exit(1)


# perform grid search to find the best hyperparameter value combination
def grid_search_hyperparameter_tuning(hyperparameter_dictionary, model, loss_fn, scaler, whole_dataset, scaled_anchors):
    # Get all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(*hyperparameter_dictionary.values()))
    print("all the hyperparameter combinations: ", hyperparameter_combinations)
    best_accuracy = 0.0
    best_hyperparameters = {}

    # Iterate over each hyperparameter combination
    for hyperparameters in hyperparameter_combinations:
        print("current hyperparameters: ", hyperparameters)
        # Unpack the hyperparameters
        OBJ_PRESENCE_CONFIDENCE_THRESHOLD, CONF_DIST_THRESHOLD, LEARNING_RATE = hyperparameters

        # Perform cross-validation
        avg_class_accuracy, avg_obj_accuracy, avg_no_obj_accuracy, avg_distance_accuracy = cross_validation(model,
                                                                                                            loss_fn,
                                                                                                            scaler,
                                                                                                            whole_dataset,
                                                                                                            scaled_anchors,
                                                                                                            hyperparameters)

        # Calculate the average accuracy
        avg_accuracy = (avg_class_accuracy + avg_obj_accuracy + avg_no_obj_accuracy + avg_distance_accuracy) / 4

        # Check if this combination has the highest accuracy so far
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_hyperparameters = {
                "OBJ_PRESENCE_CONFIDENCE_THRESHOLD": OBJ_PRESENCE_CONFIDENCE_THRESHOLD,
                "CONF_DIST_THRESHOLD": CONF_DIST_THRESHOLD,
                "LEARNING_RATE": LEARNING_RATE
            }

    # Print the best hyperparameter combination and its accuracy
    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Accuracy:", best_accuracy)

    return best_hyperparameters, (avg_class_accuracy, avg_obj_accuracy, avg_no_obj_accuracy, avg_distance_accuracy)


def main():
    torch.cuda.empty_cache()
    # defining the necessary components for training a YOLOv3
    # creating an instance of the model class
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    initialize_weights(model)

    # creating an instance of the Adam optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    # creating an instance of the loss/cost class
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    # The resulting scaled_anchors tensor will have the same shape as the config.ANCHORS tensor
    # but with the anchor boxes scaled according to the config.S parameter.
    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # using a custom get_loaders function to create data loaders for the training,
    # testing, and evaluation datasets. The data is loaded from CSV files which
    # contain the file paths and annotations for each image. These data loaders are
    # used later in the training loop.
    train_loader = get_loaders()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        if config.NUM_EPOCHS / 4 == epoch:
            name = 'epoch number {}.pk'.format(epoch)
            with open(name, 'wb') as file:
                pickle.dump(model, file)
                file.close()

    from PIL import Image
    transform = config.train_transforms
    img = np.array(Image.open("Dataset/images/000000.png").convert("RGB"))
    augmentations = transform(image=img)
    img = augmentations["image"]
    predict(model, img.unsqueeze(0).to(config.DEVICE))

    # train the model with the whole dataset since now we do know the performance of the model,
    # no need to split the dataset into train and test to test the model again. we can assume
    # that our model will perform at least as good as the cross validation result.
    # I think this function should return us the weights of the model!!
    train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
    with open('dist_yolo_model.pk', 'wb') as file:
        pickle.dump(model, file)
        file.close()

    # model evaluation below
    # check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)


    # this is another performance metric. it measures how accurate the alignment of bb's are.
    # pred_boxes, true_boxes = get_evaluation_bboxes(
    #     test_loader,
    #     model,
    #     iou_threshold=config.NMS_IOU_THRESH,
    #     anchors=config.ANCHORS,
    #     threshold=config.CLASS_CONF_THRESHOLD,
    # )
    # mapval = mean_average_precision(
    #     pred_boxes,
    #     true_boxes,
    #     iou_threshold=config.MAP_IOU_THRESH,
    #     box_format="midpoint",
    #     num_classes=config.NUM_CLASSES,
    # )
    # print(f"MAP: {mapval.item()}")


if __name__ == '__main__':
    main()
