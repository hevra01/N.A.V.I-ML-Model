from torch.utils.data import DataLoader
from torch.utils.data import Subset
import itertools

from dataset import YOLODataset
import config
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
import pickle

from dataset import YOLODataset
from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from cost import YoloLoss

torch.backends.cudnn.benchmark = True


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
        # move the input data (x) and the targets (y) onto the GPU device
        # specified in the configuration file (config.DEVICE)
        # scale1 = []
        # scale2 = []
        # scale3 = []
        #
        # # y is originally a list of images each with 3 scales. we need
        # # to convert it into a list of 3 scales that has images.
        # for img in y:
        #     scale1.append(img[0])
        #     scale2.append(img[1])
        #     scale3.append(img[2])
        #
        # scale1 = torch.stack(scale1)
        # scale2 = torch.stack(scale2)
        # scale3 = torch.stack(scale3)

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
        mean_loss = sum(losses) / len(losses)
        # sets the progress bar's display to show the current mean loss value
        loop.set_postfix(loss=mean_loss)


# this function gets the output of the model, which is predictions in three
# different scales, and combines them


def predict(model, img):
    bbox = []
    confidences = []
    class_ids = []

    predictions = model(img)

    for scale in predictions:
        for image in scale:
            for anchor in image:
                for gridx in anchor:
                    for gridy in gridx:
                        print(gridy.shape)
                        scores = gridy[:7]
                        class_id = np.argmax(scores.detach().numpy)
                        confidence = scores[class_id]
                        if confidence > 0.1:
                            center_x = int(gridy[7] * config.IMAGE_SIZE)
                            center_y = int(gridy[8] * config.IMAGE_SIZE)
                            w = int(gridy[9] * config.IMAGE_SIZE)
                            h = int(gridy[9] * config.IMAGE_SIZE)

                            x = int(center_x - (w / 2))
                            y = int(center_y - (h / 2))
                            bbox.append([x, y, w, h])
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
    result = non_max_suppression(bbox,config.NMS_IOU_THRESH,config.OBJ_PRESENCE_CONFIDENCE_THRESHOLD)
    print(result)
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

    from PIL import Image
    transform = config.train_transforms
    from yolov5.utils.dataloaders import letterbox
    img = np.array(Image.open("Dataset/images/000000.png").convert("RGB"))
    augmentations = transform(image=img)
    img = augmentations["image"]
    predict(model, img.unsqueeze(0))

    # using a custom get_loaders function to create data loaders for the training,
    # testing, and evaluation datasets. The data is loaded from CSV files which
    # contain the file paths and annotations for each image. These data loaders are
    # used later in the training loop.
    train_loader = get_loaders()
    whole_dataset = YOLODataset("Dataset/labels.txt")

    # perform grid search for hyperparameter tuning based on the hyperparameter values
    # present in the config file. if you want to change the range of values for hyperparameters,
    # please update the hyper_parameters_dictionary in config file.
    hyperparameter_dictionary = config.hyper_parameters_dictionary
    best_hyperparameter_values = grid_search_hyperparameter_tuning(hyperparameter_dictionary, model, loss_fn, scaler,
                                                                   whole_dataset, scaled_anchors)

    best_hyperparameter_values = {}
    # perform cross validation to get the average performance of the model
    # based on the best hyperparameters obtained during cross-validation
    avg_class_accuracy, avg_obj_accuracy, avg_no_obj_accuracy, avg_distance_accuracy = cross_validation(model,
                                                                                                        optimizer,
                                                                                                        loss_fn, scaler,
                                                                                                        whole_dataset,
                                                                                                        scaled_anchors,
                                                                                                        best_hyperparameter_values)

    print("\nModel performance based on cross validation:")
    print(f"Class accuracy is: {avg_class_accuracy:2f}%")
    print(f"No obj accuracy is: {avg_obj_accuracy:2f}%")
    print(f"Obj accuracy is: {avg_no_obj_accuracy:2f}%")
    print(f"Distance accuracy is: {avg_distance_accuracy:2f}%")

    # train the model with the whole dataset since now we do know the performance of the model,
    # no need to split the dataset into train and test to test the model again. we can assume
    # that our model will perform at least as good as the cross validation result.
    # I think this function should return us the weights of the model!!
    train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
    with open('dist_yolo_model.pk', 'wb') as file:
        pickle.dump(model, file)
        file.close()

    # load a previously saved model checkpoint from the specified file config.CHECKPOINT_FILE,
    # and restore the state of the model and optimizer objects so that training can continue
    # from the previously saved point.
    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
    #     )
    #
    #
    # for epoch in range(config.NUM_EPOCHS):
    #     train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

    # if config.SAVE_MODEL:
    #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

    # print(f"Currently epoch {epoch}")
    # print("On Train Eval loader:")
    # print("On Train loader:")
    # check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

    # evaluating the model's performance on the test dataset at regular intervals (every 3 epochs) during training.
    # if epoch > 0 and epoch % 3 == 0:
    #     check_class_accuracy(model, test_loader, threshold=config.OBJ_PRESENCE_CONFIDENCE_THRESHOLD,
    #                          dist_threshold=config.CONF_DIST_THRESHOLD)

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
