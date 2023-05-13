import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms import transforms

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]


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
                 transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640))])):
        if anchors is None:
            anchors = ANCHORS
        self.input_size = input_size
        self.transform = transform
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])

        with open(annotation_file, "r") as f:
            self.annotations = f.readlines()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index].strip().split(" ")
        image_path = annotation[0]
        boxes = []
        for box in annotation[1:]:
            box = box.split(",")
            box = [int(b) for b in box]
            boxes.append(box)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        targets = []
        scales = [32, 16, 8]
        for S in scales:
            target = torch.zeros((self.anchors.shape[0] // 3, self.input_size // S, self.input_size // S, 6))
            anchor_boxes = torch.zeros(self.anchors.shape[0] // 3, 4)
            for i, anchor in enumerate(ANCHORS):
                anchor_boxes[i] = [0, 0, anchor[0], anchor[1]]
            anchor_boxes = np.tile(anchor_boxes, (self.input_size//S * self.input_size//S, 1)).reshape(-1, self.anchors.shape[0], 4)
            for box in boxes:
                x_center = (box[0]+box[2])/(2*self.input_size)
                y_center = (box[1] + box[3]) / (2 * self.input_size)
                box_width = (box[2] - box[0]) / self.input_size
                box_height = (box[3] - box[1]) / self.input_size
                box_data = [x_center, y_center, box_width, box_height, 1]
                iou = bbox_iou(box_data, anchor_boxes)




        return self.transform(image), targets

        # targets = torch.zeros((3, self.input_size//32, self.input_size//32, 7))
        # for box in boxes:
        #     x, y, w, h, c, d = box
        #     box = [x / image.width, y / image.height, w / image.width, h / image.height, c, d]
        #     ious = []
        #     for scale in range(3):
        #         for i in range(self.input_size//32):
        #             for j in range(self.input_size//32):
        #                 anchor_box = torch.zeros(4)
        #                 anchor_box[2:] = torch.tensor(self.anchors[scale])
        #                 cx = (j + 0.5) / self.input_size//32
        #                 cy = (i + 0.5) / self.input_size//32
        #                 anchor_xywh = torch.zeros(4)
        #                 anchor_xywh[:2] = torch.tensor([cx, cy])
        #                 anchor_xywh[2:] = anchor_box[2:] / self.input_size
        #                 iou = bbox_iou(anchor_xywh, torch.tensor(box))
        #                 ious.append((iou, scale, i, j))
        #     best_iou, best_scale, i, j = max(ious, key=lambda x: x[0])
        #     anchor_taken = targets[best_scale, i, j, 0]
        #     if not anchor_taken:
        #         targets[best_scale, i, j, 0] = 1
        #         targets[best_scale, i, j, 1:5] = torch.tensor(box[:4])
        #         targets[best_scale, i, j, 5] = box[4]
        #         targets[best_scale, i, j, 6] = box[5]
        #     else:
        #         if best_iou > 0.5:
        #             targets[best_scale, i, j, 0] = -1
        #         else:
        #             pass


def bbox_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-16)
    return iou

def collate_fn(batch):
    images = []
    targets = []
    # batch gets prepared correctly
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return torch.stack(images, dim=0), targets
