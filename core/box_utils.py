import torch


# 将 center 格式坐标转为 minmax 格式坐标
def to_minmax_coordinates(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)


# 将 minmax 格式坐标转成 center 格式坐标
def to_center_coordinates(boxes):
    return torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), dim=1)


# 计算两组 boxes 的 iou 等，boxes 采用 minmax 格式坐标
def jaccard(boxes_a, boxes_b, mode='iou', eps=1e-6):
    # 通过 broadcast 机制将坐标拓展为 A * B * 2
    lt = torch.max(boxes_a[:, :2].unsqueeze(dim=1), boxes_b[:, :2])
    rb = torch.min(boxes_a[:, 2:].unsqueeze(dim=1), boxes_b[:, 2:])
    inter = torch.clamp(rb - lt, min=0)

    area_inter = torch.prod(inter, dim=2)
    area_a = torch.prod(boxes_a[:, 2:] - boxes_a[:, :2], dim=1)
    area_b = torch.prod(boxes_b[:, 2:] - boxes_b[:, :2], dim=1)

    if mode == 'iou':
        return area_inter / torch.max(area_a.unsqueeze(dim=1) + area_b - area_inter, torch.tensor(eps))
    elif mode == 'iof':
        return area_inter / torch.max(torch.min(area_a.unsqueeze(dim=1), area_b), torch.tensor(eps))
    elif mode == 'ioa':
        return area_inter / torch.max(area_a.unsqueeze(dim=1), torch.tensor(eps))
    else:
        return area_inter / torch.max(area_b, torch.tensor(eps))


def encode(boxes, anchors, variances):
    # boxes 为 minmax 坐标格式
    # anchors 为 center 坐标格式
    # variances 分别为中心点方差、长宽方差
    boxes = to_center_coordinates(boxes)
    g_cxcy = (boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:] / variances[0]
    g_wh = torch.log(boxes[:, 2:] / anchors[:, 2:]) / variances[1]
    return torch.cat((g_cxcy, g_wh), dim=1)


def decode(loc, anchors, variances):
    # loc 为预测的结果
    # anchors 为 center 坐标格式
    # variances 分别为中心点方差、长宽方差
    p_cxcy = loc[:, :2] * variances[0] * anchors[:, 2:] + anchors[:, :2]
    p_wh = torch.exp(loc[:, 2:] * variances[1]) * anchors[:, 2:]
    boxes = torch.cat((p_cxcy, p_wh), dim=1)
    return to_minmax_coordinates(boxes)


def match(gt_labels, gt_boxes, gt_difficults, anchors, threshold, variances):
    # 每个 ground_truth 与每个 anchor 的 iou
    overlaps = jaccard(gt_boxes, to_minmax_coordinates(anchors), mode='iou')
    # 与 ground_truth 最大 iou 的 anchor
    best_anchor_per_gt_iou, best_anchor_per_gt_idx = overlaps.max(dim=1)
    # 与 anchor 最大 iou 的 ground_truth
    best_gt_per_anchor_iou, best_gt_per_anchor_idx = overlaps.max(dim=0)

    # 保证与 ground_truth 最大 iou 的 anchor 的 target 为该 ground_truth
    for gt_idx, anchor_idx in enumerate(best_anchor_per_gt_idx):
        best_gt_per_anchor_idx[anchor_idx] = gt_idx
    # 忽略掉 ground_truth 与 anchor 最大 iou 都小于阈值的 ground_truth
    best_anchor_per_gt_filter = best_anchor_per_gt_idx[best_anchor_per_gt_iou >= 0.2]
    best_gt_per_anchor_iou = best_gt_per_anchor_iou.index_fill(dim=0, index=best_anchor_per_gt_idx, value=2)

    # 得到每个 anchor 对应的 target 标签，这里拷贝一次，防止对 gt 的篡改
    target_labels = gt_labels[best_gt_per_anchor_idx].clone()
    # 背景的标签必须为 0
    target_labels[best_gt_per_anchor_iou < threshold] = 0

    # 得到每个 anchor 对应的 target 位置
    target_boxes = gt_boxes[best_gt_per_anchor_idx]
    target_locations = encode(target_boxes, anchors, variances)

    # 得到每个 anchor 对应的 gt 的难易程度
    target_difficults = gt_difficults[best_gt_per_anchor_idx]

    return target_labels, target_locations, target_difficults


def nms(boxes, scores, threshold, mode='iou', topk=-1, candidates_size=-1):
    keep = []

    # 按照置信度降序排序
    _, order = torch.sort(scores, dim=0, descending=True)
    if candidates_size == -1:
        candidates_size = order.numel()
    order = order[:candidates_size]

    while order.numel() > 0:
        # 概率值最高的放进 keep 列表中
        keep.append(order[0].item())
        if order.numel() == 1 or 0 < topk == len(keep):
            break

        # 计算剩下 boxes 与当前 box 的 iou
        current_box = boxes[order[0]]
        rest_boxes = boxes[order[1:]]
        overlaps = jaccard(current_box.unsqueeze(dim=0), rest_boxes, mode=mode)

        # 将 iou 大于阈值的 boxes 清除掉，这里采用在 order 中清除，达到类似的效果
        mask = overlaps.squeeze() <= threshold
        order = order[1:][mask]

    return torch.tensor(keep)


class MatchStrategy:
    def __init__(self, anchors, threshold, variances):
        self.anchors = anchors
        self.threshold = threshold
        self.variances = variances

    def __call__(self, gt_labels, gt_boxes, gt_difficults):
        return match(gt_labels, gt_boxes, gt_difficults, self.anchors, self.threshold, self.variances)
