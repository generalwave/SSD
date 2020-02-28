import torch
import torch.nn.functional as functional


class MultiBoxLoss:
    def __init__(self, negative_positive_ratio, location_weight):
        self.negative_positive_ratio = negative_positive_ratio
        self.location_weight = location_weight

    def _hard_negative_mining(self, p_confidence, gt_labels, positive_mask):
        batch_size = p_confidence.size(0)
        num_classes = p_confidence.size(2)

        # 得到所需负样本的数量
        positive_num = torch.sum(positive_mask, dim=1, keepdim=True)
        negative_num = self.negative_positive_ratio * positive_num

        with torch.no_grad():
            # 计算分类损失，然后将正例的分类损失设为 0
            loss = functional.cross_entropy(p_confidence.view(-1, num_classes), gt_labels.view(-1), reduction='none')
            loss = loss.view(batch_size, -1)
            loss[positive_mask] = 0
            # 先对负样本 loss 进行排序，得到 idx 的排序后，那么再对 idx 排序，得到的 idx 的 idx 就是原始的 rank
            _, loss_idx = torch.sort(loss, dim=1, descending=True)
            _, loss_rank = torch.sort(loss_idx, dim=1)
            negative_mask = loss_rank < negative_num

        return negative_mask

    def __call__(self, p_confidences, p_locations, gt_labels, gt_boxes):
        # 获取正样本位置，保证背景的标签为 0
        positive_mask = gt_labels > 0

        # 下面需要进行负样本的 OHEM，最关键为获取负例损失的排序
        negative_mask = self._hard_negative_mining(p_confidences, gt_labels, positive_mask)

        # 计算分类损失，选择器选择之后维度会坍缩
        conf_filter = positive_mask | negative_mask
        conf_p = p_confidences[conf_filter]
        conf_gt = gt_labels[conf_filter]
        loss_c = functional.cross_entropy(conf_p, conf_gt, reduction="sum")

        # 计算位置损失
        loc_p = p_locations[positive_mask]
        loc_gt = gt_boxes[positive_mask]
        loss_l = functional.smooth_l1_loss(loc_p, loc_gt, reduction="sum")

        # 计算正样本的总个数
        n = loc_gt.size(0)

        return loss_c / n, self.location_weight * loss_l / n
