import torch
import numpy as np


def iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    ltx = np.max([boxes1[0], boxes2[0]], 0)
    lty = np.max([boxes1[1], boxes2[1]], 0)
    rbx = np.min([boxes1[2], boxes2[2]], 0)
    rby = np.min([boxes1[3], boxes2[3]], 0)

    w = (rbx - ltx)
    h = (rby - lty)
    inter = w*h  # 相交面积
    ovr = area1 + area2 - inter
    iou = inter/ovr
    return iou


def iot(boxes1, boxes2):
    """
    用于计算proposal和标注框相交面积和标注框面积的比值，以确定proposal的标签
        Arguments:
            boxes1:proposal
            boxes2:标注框
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    ltx = np.max([boxes1[0], boxes2[0]], 0)
    lty = np.max([boxes1[1], boxes2[1]], 0)
    rbx = np.min([boxes1[2], boxes2[2]], 0)
    rby = np.min([boxes1[3], boxes2[3]], 0)

    w = (rbx - ltx)
    h = (rby - lty)
    inter = w*h  # 相交面积
    iou = inter/area2
    return iou


def nms(boxes, scores, thresh=0.3):
    order = np.argsort(-1*scores)  # scores降序排列,order元素对应scores里面的索引
    keep = []
    j = 0
    while len(order) > 0:
        keep.append(boxes[order[0]])
        order = np.delete(order, 0)
        toDelete = []  # 存储将要被删除的索引

        for i in range(len(order)):
            if iou(boxes[order[i]], keep[j]) > thresh:
                toDelete.append(i)

        order = np.delete(order, toDelete)
        scores = np.delete(scores, toDelete)
        j = j+1
    return keep, scores


if __name__ == '__main__':
    box1 = [200, 200, 300, 300]
    box2 = [200, 200, 300, 300]
    scores = np.array([1, 0.9])
    print(nms([box1, box2], scores, 0))
