import config
import torch
import torch.optim as optim
from sklearn.model_selection import KFold

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

def cross_validation(model, dataset):
    train_dataset = YOLODataset("Dataset/labels.txt")

    # Define the number of folds for cross-validation
    k_folds = 5

    # Split your dataset into K folds
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Loop over the K folds
    for train_index, val_index in kfold.split(train_dataset):
        # Get the training and validation sets for the current fold
        train_set = dataset[train_index]
        val_set = dataset[val_index]

        # Train your YOLOv3 model on the training set
        model.train(train_set)

        # Evaluate the model on the validation set
        metrics = check_class_accuracy(model, loader, threshold, dist_threshold)

        # Print the evaluation metrics for the current fold
        print("Fold metrics:", metrics)

    # Calculate the average metrics across all folds
    avg_metrics = calculate_average_metrics()

    # Print the average metrics
    print("Average metrics:", avg_metrics)


def calculate_average_metrics(metrics_list):
    # Convert metrics list to numpy array for easy calculation
    metrics_array = np.array(metrics_list)

    # Calculate average metrics across all folds
    avg_metrics = np.mean(metrics_array, axis=0)

    return avg_metrics



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

    # using a custom get_loaders function to create data loaders for the training,
    # testing, and evaluation datasets. The data is loaded from CSV files which
    # contain the file paths and annotations for each image. These data loaders are
    # used later in the training loop.
    train_loader= get_loaders()

    # load a previously saved model checkpoint from the specified file config.CHECKPOINT_FILE,
    # and restore the state of the model and optimizer objects so that training can continue
    # from the previously saved point.
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    # The resulting scaled_anchors tensor will have the same shape as the config.ANCHORS tensor
    # but with the anchor boxes scaled according to the config.S parameter.
    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

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
