import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

# change: look into augmentation before training for distance estimation.
# for that, we will need to find the weights responsible from distanc estimation
# and freeze them. If we can't find them, then we shouldn't do augmentation
# from albumentations.pytorch import ToTensorV2

DATASET = 'KITTI'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 4
IMAGE_SIZE = 640  # change based on kitti
NUM_CLASSES = 7  # change: this will depend on our dataset. changed based on kitti
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200 #based on dist-yolo
# this is likely to change. used to calculate the accuracy of dist estimation. if the difference of the estimated
# distance is below this threshold than assume for a correct prediction. if the difference is more than the threshold
# calculate it as wrong prediction.
CONF_DIST_THRESHOLD = 1
OBJ_PRESENCE_CONFIDENCE_THRESHOLD = 0.45 # Filters low probability detections.
NMS_THRESHOLD = 0.6 # To remove overlapping bounding boxes
CLASS_CONFIDENCE_THRESHOLD = 0.5 # To filter low probability class scores.
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

# hyper-parameters
hyper_parameters_dictionary = {
    "OBJ_PRESENCE_CONFIDENCE_THRESHOLD": [0.35, 0.4, 0.45, 0.5],
    "CONF_DIST_THRESHOLD": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5], # values are given in meters
    "LEARNING_RATE": [0.0001, 0.001, 0.005, 0.01]
}

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
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Affine(shear=15, p=0.5),
            ],
            p=1.0,
        ),

        A.HorizontalFlip(p=0.5),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ]
)
