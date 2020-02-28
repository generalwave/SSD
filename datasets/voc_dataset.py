from torch.utils import data
import os
from xml.etree import ElementTree
import cv2
import torch
import numpy as np
from .voc_config import CLASSES_NAME


# 通过 xml 获取 object 的标注
class VOCAnnotationTransform:
    def __init__(self, keep_difficult=False):
        self.class_to_idx = dict(zip(CLASSES_NAME, range(len(CLASSES_NAME))))
        self.keep_difficult = keep_difficult

    def __call__(self, anno_path):
        labels = []
        boxes = []
        difficults = []
        root = ElementTree.parse(anno_path).getroot()

        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue

            name = obj.find('name').text.lower().strip()
            labels.append(self.class_to_idx[name])

            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for pt in pts:
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            boxes.append(bndbox)

            difficults.append(difficult)

        return np.array(labels), np.array(boxes).astype(np.float32), np.array(difficults).astype(np.bool)


# 融合多个VOC的数据集
class VOCDataset(data.Dataset):
    def __init__(self, root, image_sets, transform, match_strategy, keep_difficult=False):
        self.root = os.path.join(root, 'VOCdevkit')
        self.image_ids = self._get_image_ids(image_sets)
        self.transform = transform
        self.match_strategy = match_strategy
        self.anno_transform = VOCAnnotationTransform(keep_difficult)

    def _get_image_ids(self, image_sets):
        ids = []
        for year, name in image_sets:
            year = 'VOC' + year
            directory = os.path.join(self.root, year)
            for line in open(os.path.join(directory, 'ImageSets', 'Main', name + '.txt')):
                ids.append([year, line.strip()])
        return ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        anno_path = os.path.join(self.root, image_id[0], 'Annotations', image_id[1] + '.xml')
        image_path = os.path.join(self.root, image_id[0], 'JPEGImages', image_id[1] + '.jpg')

        labels, boxes, difficults = self.anno_transform(anno_path)
        image = cv2.imread(image_path)
        shape = torch.tensor(image.shape[:2])

        # 进行数据增强
        image, labels, boxes = self.transform(image, labels, boxes)

        # BGR 转 RGB
        image = image[:, :, (2, 1, 0)]
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        labels = torch.from_numpy(labels)
        boxes = torch.from_numpy(boxes)
        difficults = torch.from_numpy(difficults)

        if self.match_strategy is not None:
            labels, boxes, difficults = self.match_strategy(labels, boxes, difficults)

        return image, labels, boxes, difficults, shape
