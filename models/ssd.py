import torch.nn as nn
import torch
import torch.nn.functional as functional


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


# 对通道进行规整处理
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.empty(n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


# 参考 torchvision.models.vgg 中的代码
def _vgg(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


# 按照论文中增加到 vgg 网络后的层
def _extras(cfg, in_channels):
    layers = []
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            kernel_size = (1, 3)[flag]
            flag = not flag
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=kernel_size, stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=kernel_size)]
        in_channels = v
    return layers


# 最后进行检测的所有层
def _multibox(base, extras, cfg, num_classes):
    conf_layers = []
    loc_layers = []

    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        conf_layers += [nn.Conv2d(base[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(base[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]

    for k, v in enumerate(extras[1::2], 2):
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]

    return conf_layers, loc_layers


class SSD(nn.Module):
    def __init__(self, base, extras, head, num_classes, phase):
        super(SSD, self).__init__()
        self.base = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.conf = nn.ModuleList(head[0])
        self.loc = nn.ModuleList(head[1])
        self.num_classes = num_classes
        self.phase = phase

        self.apply(weights_init)

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        # 取到 conv4_3 relu 结果
        for k in range(23):
            x = self.base[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # 取到 fc7 结果
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        sources.append(x)

        # 取到额外层的结果
        for k, v in enumerate(self.extras):
            x = functional.relu(v(x), inplace=True)
            if k & 1 == 1:
                sources.append(x)

        # 进行预测
        for (x, c, l) in zip(sources, self.conf, self.loc):
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())

        conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], dim=1)
        loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], dim=1)

        if self.phase == "test":
            conf = functional.softmax(conf, dim=-1)

        return conf, loc


# 当前只支持 SSD300
vgg_cfg = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
}
extras_cfg = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
}
# 每层 feature_map 的每个 cell 包含的待检 anchor 个数
multibox_cfg = {
    '300': [4, 6, 6, 6, 4, 4]
}


def build_ssd(size=300, num_classes=21, phase='train'):
    base = _vgg(vgg_cfg[str(size)], in_channels=3, batch_norm=False)
    extras = _extras(extras_cfg[str(size)], 1024)
    head = _multibox(base, extras, multibox_cfg[str(size)], num_classes)

    model = SSD(base, extras, head, num_classes, phase)

    return model
