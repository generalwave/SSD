import torch.utils.data as data
from tqdm import tqdm
from time import time
import torch
import os
from core.multibox_loss import MultiBoxLoss
from core.anchor_generator import AnchorGenerator
from core.box_utils import MatchStrategy
from datasets.augmentation import DataAugmentation
from torch.optim.lr_scheduler import MultiStepLR
from models.ssd import build_ssd
from datasets.voc_dataset import VOCDataset
from datasets.voc_config import CONFIG


def get_voc_data_loader(mode='train'):
    # 训练配置参数
    batch_size = CONFIG['batch_size']
    thread_num = CONFIG['thread_num']
    # AnchorGenerator 的参数
    image_size = CONFIG['image_size']
    feature_maps_size = CONFIG['feature_maps_size']
    strides = CONFIG['strides']
    scales = CONFIG['scales']
    aspect_ratios_list = CONFIG['aspect_ratios_list']
    # Dataset 参数
    if mode == 'train':
        root = CONFIG['root']
        image_sets = CONFIG['image_sets']
    else:
        return None
    threshold = CONFIG['threshold']
    variance = CONFIG['variance']
    # DataAugmentation 参数
    bgr_mean = CONFIG['bgr_mean']

    anchor_generator = AnchorGenerator(image_size, feature_maps_size, strides, scales, aspect_ratios_list)
    anchors = anchor_generator()
    match_strategy = MatchStrategy(anchors, threshold, variance)

    augmentation = DataAugmentation(image_size, bgr_mean)

    dataset = VOCDataset(root, image_sets, augmentation, match_strategy)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=thread_num, drop_last=True)

    return dataloader


def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()

    total_num = 0
    total_loss = 0
    total_c_loss = 0
    total_l_loss = 0

    for images, labels, boxes, _, _ in tqdm(loader):
        batch_size = images.size(0)
        images = images.to(device)
        labels = labels.to(device)
        boxes = boxes.to(device)

        confidences, locations = model(images)

        optimizer.zero_grad()
        loss_c, loss_l = criterion(confidences, locations, labels, boxes)
        loss = loss_c + loss_l
        loss.backward()
        optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        total_c_loss += loss_c.item() * batch_size
        total_l_loss += loss_l.item() * batch_size

    if total_num != 0:
        total_loss = total_loss / total_num
        total_c_loss = total_c_loss / total_num
        total_l_loss = total_l_loss / total_num

    return total_loss, total_c_loss, total_l_loss


def val_one_epoch(model, device, loader,  criterion):
    model.eval()

    total_num = 0
    total_loss = 0
    total_c_loss = 0
    total_l_loss = 0

    with torch.no_grad():
        for images, labels, boxes, _, _ in loader:
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            boxes = boxes.to(device)

            confidences, locations = model(images)
            loss_c, loss_l = criterion(confidences, locations, labels, boxes)
            loss = loss_c + loss_l

            total_num += batch_size
            total_loss += loss.item() * batch_size
            total_c_loss += loss_c.item() * batch_size
            total_l_loss += loss_l.item() * batch_size

    if total_num != 0:
        total_loss = total_loss / total_num
        total_c_loss = total_c_loss / total_num
        total_l_loss = total_l_loss / total_num

    return total_loss, total_c_loss, total_l_loss


def train(model, device):
    # 训练配置参数
    max_epoch = CONFIG['max_epoch']
    lr = CONFIG['lr']
    momentum = CONFIG['momentum']
    weight_decay = CONFIG['weight_decay']
    # 学习率调整参数
    milestones = CONFIG['milestones']
    gamma = CONFIG['gamma']
    # MultiBoxLoss 参数
    negative_positive_ratio = CONFIG["negative_positive_ratio"]
    location_weight = CONFIG["location_weight"]
    # 模型保存路径
    save_directory = CONFIG['save_directory']
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(negative_positive_ratio, location_weight)
    scheduler = MultiStepLR(optimizer, milestones, gamma)

    train_loader = get_voc_data_loader('train')
    val_loader = get_voc_data_loader('val')

    for i in range(max_epoch):
        start = time()
        t_loss, t_c_loss, t_l_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
        if val_loader is not None:
            v_loss, v_c_loss, v_l_loss = val_one_epoch(model, device, val_loader, criterion)
        else:
            v_loss, v_c_loss, v_l_loss = t_loss, t_c_loss, t_l_loss
        end = time()

        scheduler.step(i)

        msg = 't_loss:%f\tt_c_loss:%f\tt_l_loss:%f' % (t_loss, t_c_loss, t_l_loss)
        msg += '\tv_loss:%f\tv_c_loss:%f\tv_l_loss:%f' % (v_loss, v_c_loss, v_l_loss)
        msg += '\ttime:%f\tepoch:%d' % (end - start, i)
        print(msg)

        params = model.state_dict()
        save_path = os.path.join(save_directory, 'SSD_epoch_' + str(i) + '_' + str(v_loss) + '.pth')
        torch.save(params, save_path)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = CONFIG['image_size']
    num_classes = CONFIG['num_classes']
    model = build_ssd(image_size, num_classes)
    model.to(device)

    pretrain_model = CONFIG['pretrain_model']
    vgg_pretrain = CONFIG['vgg_pretrain']
    if pretrain_model is not None:
        model.load_state_dict(torch.load(pretrain_model, map_location=device))
    elif vgg_pretrain is not None:
        model.base.load_state_dict(torch.load(vgg_pretrain))

    train(model, device)


if __name__ == '__main__':
    main()
