import config
import torch
import torch.optim as optim

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
        # specified in the configuration file (config.DEVICE).
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        # performing forward pass and loss computation in a mixed precision setting using PyTorch's AMP feature.
        with torch.cuda.amp.autocast():
            out = model(x)
            # change: this is kinda weird because in the dataset I don't think there are moer than one label per image
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
