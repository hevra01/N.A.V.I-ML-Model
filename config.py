# import albumentations as A
import torch

# change: look into augmentation before training for distance estimation.
# for that, we will need to find the weights responsible from distanc estimation
# and freeze them. If we can't find them, then we shouldn't do augmentation
# from albumentations.pytorch import ToTensorV2
from utils import seed_everything

DATASET = 'PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 8
IMAGE_SIZE = 416  # change based on kitti
NUM_CLASSES = 7  # change: this will depend on our dataset. changed based on kitti
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
# this is likely to change. used to calculate the accuracy of dist estimation. if the difference of the estimated
# distance is below this threshold than assume for a correct prediction. if the difference is more than the threshold
# calculate it as wrong prediction.
CONF_DIST_THRESHOLD = 1.5
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

# change: these can change based on our dataset. check notion cng 492/Dataset/anchors
# we can check which anchor values dist-yolo used.
# dist-yolo anchors: 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
# maybe normalization will be needed (scale to be between (0-1))
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

scale = 1.1

# change classes based on kitti
KITTI_CLASSES = [
    "car",
    "cyclist",
    "pedestrian",
    "van",
    "truck",
    "tram",
    "person sitting"
]
