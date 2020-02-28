# SSD: Single Shot MultiBox Detector
利用pytorch重现 [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)  论文效果，这里只实现对VOC数据集、SSD300*的支持。官方采用Caffe实现，详见 [here](https://github.com/weiliu89/caffe/tree/ssd)

## 数据集

### VOC2007
[trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
[test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)

### VOC2012
[trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

## 训练
- 解压VOC数据集到同一个目录，并修改 datasets/voc_config.py 中的 root 为该解压目录
- 下载 [VGG16的预训练模型](https://pan.baidu.com/s/1hgUm1WHsdxywHRIN_HuIhQ) ，提取码:0u0u
- 修改 datasets/voc_config.py 中的 vgg_pretrain
- 修改 datasets/voc_config.py 中的 save_directory
- 运行 ./train.py ，注意这里坏境为 python3 + pytorch1.3

## 评估
- 修改 ./evaluation.py 中的 pretrain_model 为待评估模型
- 运行 ./evaluation.py

## 预训练模型
[预训练模型](https://pan.baidu.com/s/1GZ1wegiBHrqmcS1G0fEcVg) ，提取码:ux2g

## 表现
| Original | Ours |
|:-:|:-:|
| 77.2 % | 77.82 % |

## 参考
[ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)