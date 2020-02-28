from core.box_utils_numpy import jaccard
import cv2
import numpy as np
from numpy import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, labels, boxes):
        for t in self.transforms:
            image, labels, boxes = t(image, labels, boxes)

        return image, labels, boxes


class ConvertFromInts(object):
    def __call__(self, image, labels, boxes):
        return image.astype(np.float32), labels, boxes


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, labels, boxes):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), labels, boxes


class ToAbsoluteCoords(object):
    def __call__(self, image, labels, boxes):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, labels, boxes


class ToPercentCoords(object):
    def __call__(self, image, labels, boxes):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, labels, boxes


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, labels, boxes):
        image = cv2.resize(image, (self.size, self.size))
        return image, labels, boxes


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, labels, boxes):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, labels, boxes


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, labels, boxes):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        return image, labels, boxes


class RandomLightingNoise(object):
    def __init__(self):
        self.permutes = ((0, 1, 2), (0, 2, 1),
                         (1, 0, 2), (1, 2, 0),
                         (2, 0, 1), (2, 1, 0))

    def __call__(self, image, labels, boxes):
        if random.randint(2):
            swap = self.permutes[random.randint(len(self.permutes))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)

        return image, labels, boxes


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, labels, boxes):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError

        return image, labels, boxes


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, labels, boxes):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha

        return image, labels, boxes


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, labels, boxes):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta

        return image, labels, boxes


class RandomSampleCrop(object):
    def __init__(self):
        # 采样选项，格式为元组(最小交并比，最大交并比)或者None
        # None代表整个图片，元组中的None代表没有限制
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def __call__(self, image, labels, boxes):
        height, width, _ = image.shape

        while True:
            # 随机选择一个采样方式
            option = random.choice(self.sample_options)
            if option is None:
                return image, labels, boxes

            min_iou, max_iou = option
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # 最多进行50次尝试
            for _ in range(50):
                w = random.randint(int(0.3 * width), width + 1)
                h = random.randint(int(0.3 * height), height + 1)
                # 限制宽高比在 [0.5, 2] 以内
                if h / w < 0.5 or h / w > 2:
                    continue
                left = random.randint(width - w + 1)
                top = random.randint(height - h + 1)
                roi = np.array([left, top, left + w, top + h])

                # 计算裁剪图片框和 ground truth 之间的交并比，使得交并比符合当前采样条件
                overlap = jaccard(roi[np.newaxis], boxes, mode='iou')
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # 保证 ground truth 的中心点在裁剪图片中
                cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                mask = np.all(np.logical_and(roi[:2] <= cxcy, cxcy < roi[2:]), axis=1)
                if not np.any(mask):
                    continue

                # 防止对原数据的更改
                image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
                labels_t = labels[mask]
                boxes_t = boxes[mask]

                # 将针对原图的坐标，转换为针对候选框的坐标，预测点进行截断
                boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
                boxes_t[:, :2] -= roi[:2]
                boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
                boxes_t[:, 2:] -= roi[:2]

                return image_t, labels_t, boxes_t


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, labels, boxes):
        if random.randint(2):
            return image, labels, boxes

        height, width, channels = image.shape

        scale = random.uniform(1, 4)
        w = int(width * scale)
        h = int(height * scale)
        left = random.randint(w - width + 1)
        top = random.randint(h - height + 1)

        expand_image = np.zeros((h, w, channels), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[top:top+height, left:left+width] = image

        image = expand_image

        boxes = boxes
        boxes[:, :2] += (left, top)
        boxes[:, 2:] += (left, top)

        return image, labels, boxes


class RandomMirror(object):
    def __call__(self, image, labels, boxes):
        _, width, _ = image.shape

        if random.randint(2):
            image = image[:, ::-1]
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, labels, boxes


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, labels, boxes):
        image, labels, boxes = self.rand_brightness(image, labels, boxes)

        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        image, labels, boxes = distort(image, labels, boxes)

        return self.rand_light_noise(image, labels, boxes)


class DataAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            # ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, image, labels, boxes):
        image, labels, boxes = self.augment(image, labels, boxes)
        return image, labels, boxes


class BaseAugmentation(object):
    def __init__(self, size, mean=(104, 117, 123)):
        self.size = size
        self.mean = mean
        self.augment = Compose([
            ConvertFromInts(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, image, labels, boxes):
        image, labels, boxes = self.augment(image, labels, boxes)
        return image, labels, boxes
