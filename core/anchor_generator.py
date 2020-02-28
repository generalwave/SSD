from itertools import product
from math import sqrt
import torch


# 生成 anchor，也就是 prior box
class AnchorGenerator:
    def __init__(self, image_size, feature_maps_size, strides, scales, aspect_ratios_list):
        self.image_size = image_size
        self.feature_maps_size = feature_maps_size
        self.strides = strides
        self.scales = scales
        self.aspect_ratios_list = aspect_ratios_list

    def __call__(self):
        anchors = []

        # 遍历每张特征图
        for idx in range(len(self.feature_maps_size)):
            feature_map_size = self.feature_maps_size[idx]
            stride = self.strides[idx]
            min_scale = self.scales[idx]
            max_scale = self.scales[idx + 1]
            aspect_ratios = self.aspect_ratios_list[idx]

            # 遍历一张特征上的每个 cell
            for y, x in product(range(feature_map_size), repeat=2):
                cx = (x + 0.5) * stride / self.image_size
                cy = (y + 0.5) * stride / self.image_size

                # 遍历每种横纵比
                for aspect_ratio in aspect_ratios:
                    if aspect_ratio == 1:
                        w = h = sqrt(min_scale * max_scale) / self.image_size
                        anchors.append([cx, cy, w, h])
                    w = min_scale * sqrt(aspect_ratio) / self.image_size
                    h = min_scale / sqrt(aspect_ratio) / self.image_size
                    anchors.append([cx, cy, w, h])

        return torch.tensor(anchors)
