# utility functions

import numpy as np

import torch

from typing import Tuple


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def bbox_yolo(box_points: Tuple[int, int, int, int], img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    # last answer on: https://stackoverflow.com/questions/49122356/bounding-boxes-for-yolo
    # see also: https://github.com/ultralytics/yolov3/issues/1543#issuecomment-721532206

    left, top, box_width, box_height = box_points
    # top, left, box_height, box_width = box_points
    height, width = img_shape

    x = (left + (box_width / 2)) / width
    y = (top + (box_height / 2)) / height
    w = box_width / width
    h = box_height / height
    return x, y, w, h
