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

    def test_intersection_over_union(self):
        # Test case 1: identical boxes
        boxes_preds = torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5]])
        boxes_labels = torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5]])
        expected_iou = torch.tensor([1.0, 1.0])
        iou = intersection_over_union(boxes_preds, boxes_labels, box_format="corners")
        assert torch.allclose(iou, expected_iou, rtol=1e-03, atol=1e-05)

        # Test case 2: partially overlapping boxes
        boxes_preds = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.5, 1.5]])
        boxes_labels = torch.tensor([[0.5, 0.5, 1.5, 1.5], [0.0, 0.0, 1.0, 1.0]])
        expected_iou = torch.tensor([0.25, 0.25])
        iou = intersection_over_union(boxes_preds, boxes_labels, box_format="corners")
        assert torch.allclose(iou, expected_iou, rtol=1e-03, atol=1e-05)

        # Test case 3: non-overlapping boxes
        boxes_preds = torch.tensor([[0.0, 0.0, 0.5, 0.5], [1.0, 1.0, 1.5, 1.5]])
        boxes_labels = torch.tensor([[0.5, 0.5, 1.0, 1.0], [1.5, 1.5, 2.0, 2.0]])
        expected_iou = torch.tensor([0.0, 0.0])
        iou = intersection_over_union(boxes_preds, boxes_labels, box_format="corners")
        assert torch.allclose(iou, expected_iou, rtol=1e-03, atol=1e-05)
