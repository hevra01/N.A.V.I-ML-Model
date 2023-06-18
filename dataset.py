import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from yolov5.utils.dataloaders import letterbox

import config


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2, _, _ = box2

    # Calculate the area of intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)

    # Calculate the area of both bounding boxes
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # Calculate the IoU
    iou = inter_area / float(b1_area + b2_area - inter_area)
    return iou


class YOLODataset(Dataset):

    def __init__(self, annotation_file, anchors=None, input_size=640,
                 transform=config.train_transforms):
        self.anchors = anchors or [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        ]
        self.input_size = input_size
        self.transform = transform

        with open(annotation_file, "r") as f:
            self.annotations = f.readlines()

    def __len__(self):
        return len(self.annotations)

    def load_image_and_boxes(self, index):
        # Get image path and annotations for the given index
        annotation = self.annotations[index].strip().split(" ")
        img_path = annotation[0]
        img = np.array(Image.open(img_path).convert("RGB"))

        img_width, img_height = img.shape[:2]

        boxes = []

        for box in annotation[1:]:
            box = box.split(",")
            box = [int(b) for b in box]
            x1, y1, x2, y2, cls, dist = [float(x) for x in box]
            box[0] = x1 / img_width
            box[1] = y1 / img_height
            box[2] = x2 / img_width
            box[3] = y2 / img_height
            boxes.append(box)
        img, _, _ = letterbox(img, (self.input_size, self.input_size))
        return img, boxes

    def __getitem__(self, index):
        img, boxes = self.load_image_and_boxes(index)
        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations["image"]
        targets_out = []
        scales = [32, 16, 8]
        for i, scale in enumerate(scales):
            grid_size = self.input_size // scale
            num_anchors = len(self.anchors[i])

            targets_scale = torch.zeros((num_anchors, grid_size, grid_size, 7))
            anchor_taken = torch.zeros((num_anchors, grid_size, grid_size))
            for box in boxes:
                box_cx, box_cy, box_w, box_h, box_class, box_dist = box
                box_w *= self.input_size
                box_h *= self.input_size
                box_cx = (box_cx * self.input_size) - 0.5 * (box_w - 1)
                box_cy = (box_cy * self.input_size) - 0.5 * (box_h - 1)

                box_x, box_y = int(min(max(box_cx // scale, 0), grid_size-1)), int(min(max(box_cy // scale, 0), grid_size-1))

                best_iou = 0.0
                best_anchor_idx = 0
                for anchor_idx, anchor in enumerate(self.anchors[i]):
                    anchor_w, anchor_h = anchor
                    anchor_box = torch.tensor([0.0, 0.0, anchor_w, anchor_h])

                    box = torch.tensor([0.0, 0.0, box_w, box_h])

                    iou = box_iou(anchor_box, box)

                    if iou > best_iou:
                        if anchor_taken[anchor_idx, box_y, box_x] == 0:
                            best_iou = iou
                            best_anchor_idx = anchor_idx
                            anchor_taken[i, box_y, box_x, best_anchor_idx] = 1
                        else:
                            targets_scale[ anchor_idx, box_y, box_x, 0] = -1  # set objectness to -1 if anchor is already taken
                            continue

                box_w = box_w / scale
                box_h = box_h / scale

                targets_scale[best_anchor_idx, box_y, box_x, 0] = 1
                targets_scale[best_anchor_idx, box_y, box_x, 1] = (box_cx // scale) - box_x
                targets_scale[best_anchor_idx, box_y, box_x, 2] = (box_cy // scale) - box_y
                targets_scale[best_anchor_idx, box_y, box_x, 3] = np.log(box_w / self.anchors[i][best_anchor_idx][0] + 1e-16) #avoids dvision by 0
                targets_scale[best_anchor_idx, box_y, box_x, 4] = np.log(box_h / self.anchors[i][best_anchor_idx][1] + 1e-16)
                targets_scale[best_anchor_idx, box_y, box_x, 5] = box_class
                targets_scale[best_anchor_idx, box_y, box_x, 6] = box_dist
            targets_out.append(targets_scale)
        return img, tuple(targets_out)


def box_iou(box1, box2):
    b1_x1, b1_y1, b1_w, b1_h = box1
    b2_x1, b2_y1, b2_w, b2_h = box2
    b1_x2, b1_y2 = b1_x1 + b1_w - 1, b1_y1 + b1_h - 1
    b2_x2, b2_y2 = b2_x1 + b2_w - 1, b2_y1 + b2_h - 1

    x_left = max(b1_x1, b2_x1)
    y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2)
    y_bottom = min(b1_y2, b2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    b1_area = b1_w * b1_h
    b2_area = b2_w * b2_h

    iou = intersection_area / float(b1_area + b2_area - intersection_area)

    return iou
