from datasets.voc_config import CONFIG, CLASSES_NAME
from models.ssd import build_ssd
from datasets.voc_dataset import VOCDataset
from datasets.augmentation import BaseAugmentation
from core.result_transform import ResultTransform
from core.anchor_generator import AnchorGenerator
from core.box_utils_numpy import jaccard
import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np


def calc_ap(recall, precision, use_07_metric=True):
    # 是否使用 07 年的 11 点均值方式计算 ap
    if use_07_metric:
        ap = 0.
        for threshold in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= threshold) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= threshold])
            ap = ap + p / 11.
    else:
        # 增加哨兵，然后算出准确率包络
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 计算 pr 曲线中，召回率变化的下标
        idx = np.where(mrec[1:] != mrec[:-1])[0]

        # 计算 pr 曲线与坐标轴所围区域的面积
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return ap


def clac_voc_ap(annotation, prediction, ovthresh=0.5):
    # 统计标注框个数，这里需要将困难样本数量去掉
    positive_num = 0
    for anno in annotation:
        positive_num += len(anno['bbox']) - anno['difficult'].sum()
        anno['b_det'] = [False] * len(anno['bbox'])

    # 将检测结果格式进行转换，主要是为了方便排序
    image_ids, confidences, bboxes = [], [], []
    for item in prediction:
        img_id = item['id']
        bbox = item['bbox']
        score = item['confidence']
        for i in range(len(score)):
            image_ids.append(img_id)
            confidences.append(score[i])
            bboxes.append(bbox[i])
    image_ids, confidences, bboxes = np.array(image_ids), np.array(confidences), np.array(bboxes)

    # 按照置信度排序
    sorted_ind = np.argsort(-confidences)
    bboxes = bboxes[sorted_ind]
    image_ids = image_ids[sorted_ind]

    # 计算 TP 和 FP，以计算出 AP 值
    detect_num = len(image_ids)
    tp = np.zeros(detect_num)
    fp = np.zeros(detect_num)
    for d in range(detect_num):
        gt_bboxes = annotation[image_ids[d]]['bbox']

        # 如果没有 ground truth，那么所有的检测都是错误的
        if gt_bboxes.size > 0:
            difficults = annotation[image_ids[d]]['difficult']
            b_dets = annotation[image_ids[d]]['b_det']
            p_bboxes = bboxes[d, :]

            overlaps = jaccard(p_bboxes[np.newaxis], gt_bboxes, mode='iou')[0]
            idxmax = np.argmax(overlaps)
            ovmax = overlaps[idxmax]

            if ovmax > ovthresh:
                # 忽略掉困难样本
                if difficults[idxmax]:
                    continue
                # gt 只允许检测出一次
                if not b_dets[idxmax]:
                    tp[d] = 1.
                    b_dets[idxmax] = True
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # 计算召回率和准确率
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / float(positive_num)
    precision = tp / np.maximum(tp + fp, 1.)
    ap = calc_ap(recall, precision)

    return recall, precision, ap


def eval_voc(annotation_result, detector_result):
    aps = []
    for cls in CLASSES_NAME[1:]:
        recall, precision, ap = clac_voc_ap(annotation_result[cls], detector_result[cls], ovthresh=0.5)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))


def get_model(device):
    image_size = CONFIG['image_size']
    num_classes = CONFIG['num_classes']
    model = build_ssd(image_size, num_classes, 'test')

    pretrain_model = '/data1/jiang.yang/output/ssd/SSD_epoch_221_1.9046952678341007.pth'
    model.load_state_dict(torch.load(pretrain_model, map_location=device))

    model.eval()

    model.to(device)

    return model


def get_voc_data_loader():
    image_size = CONFIG['image_size']
    bgr_mean = CONFIG['bgr_mean']
    augmentation = BaseAugmentation(image_size, bgr_mean)

    root = CONFIG['root']
    image_sets = [('2007', 'test')]
    dataset = VOCDataset(root, image_sets, augmentation, match_strategy=None, keep_difficult=True)

    dataloader = data.DataLoader(dataset)

    return dataloader


def get_result_transform():
    image_size = CONFIG['image_size']
    feature_maps_size = CONFIG['feature_maps_size']
    strides = CONFIG['strides']
    scales = CONFIG['scales']
    aspect_ratios_list = CONFIG['aspect_ratios_list']
    anchor_generator = AnchorGenerator(image_size, feature_maps_size, strides, scales, aspect_ratios_list)
    anchors = anchor_generator()

    variance = CONFIG['variance']
    conf_thresh = 0.01
    nms_thresh = 0.45
    top_k = 200

    result_transform = ResultTransform(anchors, variance, conf_thresh, nms_thresh, top_k)

    return result_transform


def test_voc(device, model, dataloader, result_transform):
    # 将每个类的最后结果及标注结果分别保存
    annotation_result = dict()
    detector_result = dict()
    for cls in CLASSES_NAME[1:]:
        annotation_result[cls] = []
        detector_result[cls] = []

    with torch.no_grad():
        for img_id, (images, labels, boxes, difficults, shapes) in enumerate(tqdm(dataloader)):
            # 这里强制只进行 batch_size 为 1 的操作
            batch_size = images.size(0)
            if batch_size != 1:
                print('评估时需采用 batch_size 为 1')
                break

            images = images.to(device)
            confidences, locations = model(images)
            confidences = confidences.cpu()
            locations = locations.cpu()

            result = result_transform(confidences, locations)[0]
            labels, boxes, difficults, shape = labels[0], boxes[0], difficults[0], shapes[0]

            h, w = shape[:2]
            # 因为 0 表示背景，因此这里从 1 开始
            for cid, cls in enumerate(CLASSES_NAME[1:]):
                cid += 1
                # 保存标注结果
                item = dict()
                item['bbox'] = boxes[labels == cid].numpy()
                item['difficult'] = difficults[labels == cid].numpy()
                annotation_result[cls].append(item)

                dets = result[cid]
                # 对与没有 top_k 的图，等于 0 就是指无效
                dets = dets[dets[:, 0] > 0]
                if dets.size(0) == 0:
                    continue
                dets[:, 1] *= w
                dets[:, 3] *= w
                dets[:, 2] *= h
                dets[:, 4] *= h

                # 保存计算结果
                item = dict()
                item['confidence'] = dets[:, 0].numpy()
                item['bbox'] = dets[:, 1:].numpy()
                item['id'] = img_id
                detector_result[cls].append(item)

    return annotation_result, detector_result


def evaluation_voc():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(device)

    dataloader = get_voc_data_loader()

    result_transform = get_result_transform()

    annotation_result, detector_result = test_voc(device, model, dataloader, result_transform)

    eval_voc(annotation_result, detector_result)


if __name__ == '__main__':
    evaluation_voc()
