import torch
from .box_utils import decode, nms


class ResultTransform:
    def __init__(self, anchors, variance, conf_thresh, nms_thresh, top_k):
        self.anchors = anchors
        self.variance = variance
        # 置信度门限
        self.conf_thresh = conf_thresh
        # nms 门限，控制重叠框
        self.nms_thresh = nms_thresh
        # 一张图检出非背景的最大个数
        self.top_k = top_k

    def __call__(self, confidences, locations):
        batch_size = confidences.size(0)
        num_classes = confidences.size(2)

        # 保存最后结果，按照 score + boxes 的方式保存
        result = torch.zeros(batch_size, num_classes, self.top_k, 5)

        for sample_id in range(batch_size):
            # 候选框进行解码
            sample_boxes = decode(locations[sample_id], self.anchors, self.variance)

            # 这里只循环非背景的分类，背景 id 默认为 0
            for cls_id in range(1, num_classes):
                mask = confidences[sample_id, :, cls_id] > self.conf_thresh
                scores = confidences[sample_id, mask, cls_id]
                if scores.size(0) == 0:
                    continue
                boxes = sample_boxes[mask]
                keep = nms(boxes, scores, self.nms_thresh, 'iou', self.top_k)

                result[sample_id, cls_id, :keep.size(0)] = torch.cat((scores[keep].unsqueeze(1), boxes[keep]), dim=1)

        # 只取 top_k 个非背景目标
        view_result = result.view(batch_size, -1, 5)
        _, idx = torch.sort(view_result[:, :, 0], dim=1, descending=True)
        _, rank = torch.sort(idx, dim=1)
        view_result[rank >= self.top_k] = 0

        return result
