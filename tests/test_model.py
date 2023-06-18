import pickle
import config
from utils import get_evaluation_bboxes, mean_average_precision
from utils import get_loaders

if __name__ == '__main__':
    with open('../epoch number 60.pk', 'rb') as file:
        model = pickle.load(file)
        file.close()

    test_loader = get_loaders()
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_THRESHOLD,
        anchors=config.ANCHORS,
        threshold=config.CLASS_CONFIDENCE_THRESHOLD,
    )
    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.NMS_THRESHOLD,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )
    print(f"MAP: {mapval.item()}")
