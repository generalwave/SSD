
CLASSES_NAME = (
    '__background__',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


CONFIG = {
    # 训练配置参数
    'max_epoch': 233,
    'lr': 1e-3,
    'milestones': [155, 194],
    'gamma': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'batch_size': 32,
    'thread_num': 4,

    # 将 anchor 设置为正例的 iou 门限
    'threshold': 0.5,
    # 中心点方差和长宽方差
    'variance': [0.1, 0.2],
    # 负样本与正样本数量比例
    'negative_positive_ratio': 3,
    # 位置回归损失权重
    'location_weight': 1,

    # 模型进行分类的数目
    'num_classes': len(CLASSES_NAME),
    # 数据集中各通道的均值
    'bgr_mean': (104, 117, 123),

    # 训练输入图片大小
    'image_size': 300,
    # 特征图的大小
    'feature_maps_size': [38, 19, 10, 5, 3, 1],
    # 各特征图的步进
    'strides': [8, 16, 32, 64, 100, 300],
    # anchor 对应的尺度
    'scales': [30, 60, 111, 162, 213, 264, 315],
    # anchor 的宽高比
    'aspect_ratios_list': [[1, 2, 1/2],
                           [1, 2, 1/2, 3, 1/3],
                           [1, 2, 1/2, 3, 1/3],
                           [1, 2, 1/2, 3, 1/3],
                           [1, 2, 1/2],
                           [1, 2, 1/2]],

    # 训练集目录
    'root': '/data1/jiang.yang/output/voc',
    # 训练集数据
    'image_sets': [('2007', 'trainval'), ('2012', 'trainval')],

    # 模型保存位置
    'save_directory': '/data1/jiang.yang/output/ssd',
    # 预训练模型
    'pretrain_model': None,
    # vgg 预训练模型
    'vgg_pretrain': '/data1/jiang.yang/pretrain/vgg16_reducedfc.pth',
}
