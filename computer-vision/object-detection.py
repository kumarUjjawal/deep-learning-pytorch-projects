# Object Detection and Bounding Boxes

from d2l import torch as d2l
import torch

# Bounding Box

def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:,0]