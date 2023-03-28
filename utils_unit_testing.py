import unittest
import torch
from utils import intersection_over_union, iou_width_height

class TestUtilsFunctions(unittest.TestCase):

    def test_iou_width_height(self):
        # Test case where both boxes are the same
        boxes1 = torch.tensor([[3, 3], [4, 4]])
        boxes2 = torch.tensor([[3, 3], [4, 4]])
        iou = iou_width_height(boxes1, boxes2)
        self.assertTrue(torch.allclose(iou, torch.tensor([1.0, 1.0])))

        # Test case where the boxes do not overlap
        boxes1 = torch.tensor([[1, 1], [2, 2]])
        boxes2 = torch.tensor([[4, 4], [5, 5]])
        iou = iou_width_height(boxes1, boxes2)
        self.assertTrue(torch.allclose(iou, torch.tensor([0.0, 0.0])))

        # Test case where the boxes partially overlap
        boxes1 = torch.tensor([[1, 1], [3, 3]])
        boxes2 = torch.tensor([[2, 2], [4, 4]])
        iou = iou_width_height(boxes1, boxes2)
        self.assertTrue(torch.allclose(iou, torch.tensor([0.1429, 0.1429]), rtol=1e-4))
