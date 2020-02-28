import numpy as np


# 计算两组 boxes 的 iou 等，boxes 采用 minmax 格式坐标，数据为 numpy 格式数据
def jaccard(boxes_a, boxes_b, mode='iou', eps=1e-6):
    # 通过 broadcast 机制将坐标拓展为 A * B * 2
    lt = np.maximum(boxes_a[:, np.newaxis, :2], boxes_b[:, :2])
    rb = np.minimum(boxes_a[:, np.newaxis, 2:], boxes_b[:, 2:])
    inter = np.maximum(rb - lt, 0)

    area_inter = np.prod(inter, axis=2)
    area_a = np.prod(boxes_a[:, 2:] - boxes_a[:, :2], axis=1)
    area_b = np.prod(boxes_b[:, 2:] - boxes_b[:, :2], axis=1)

    if mode == 'iou':
        return area_inter / np.maximum(area_a[:, np.newaxis] + area_b - area_inter, eps)
    elif mode == 'iof':
        return area_inter / np.maximum(np.minimum(area_a[:, np.newaxis], area_b), eps)
    elif mode == 'ioa':
        return area_inter / np.maximum(area_a[:, np.newaxis], eps)
    else:
        return area_inter / np.maximum(area_b, eps)
