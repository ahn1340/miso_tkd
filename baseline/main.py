import os
import math
import datetime
import time
import copy
import numpy as np
import time

import torch
#import torchsummary
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms
import torchvision.models as models
import argparse
from data_local_loader import test_data_loader, data_loader_with_split
from tqdm import tqdm

# custom modules
from model import r2plus1d_18



try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train')
    VAL_DATASET_PATH = None
except:
    IS_ON_NSML=False
    TRAIN_DATASET_PATH = os.path.join('train')
    VAL_DATASET_PATH = None


def validate(epoch, model):
    global best_acc

    model.eval()
    valid_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, samples in enumerate(val_loader):
            inputs = samples[1].to(device)
            targets = samples[2].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Validation: Epoch=%d, Loss=%.3f, Acc=%.3f' % (epoch, valid_loss / total,  correct / total))


    # Save checkpoint.
    acc = correct / total
    if acc > best_acc:
        # print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        best_acc = acc
        print('Best Acc : %.3f' % best_acc)

    if IS_ON_NSML:
        nsml.report(
            summary=True,
            step=epoch,
            epoch_total=epoch_times,
            val_loss=valid_loss / total,
            val_acc=correct / total
        )


def train(epoch, model, scheduler):
    print("Start Epoch {}".format(epoch))
    # train mode
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    clip_gradient = 40

    for batch_i, sample in enumerate(tr_loader):
        start_time = time.time()
        optimizer.zero_grad()

        inputs, targets = sample[1], sample[2]  # image shape: 3 * 224 * 224
        inputs = inputs.to(device)
        print(inputs.size())
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), clip_gradient)
            if total_norm > clip_gradient:
                print("clipping gradient: {} with coef {}".format(
                    total_norm, clip_gradient / total_norm))

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        end_time = time.time()
        duration = end_time - start_time

        print('Epoch %d Batch %d Loss=%.3f (%.3f), Acc=%.3f, Time = %.3f' % (epoch,
                                                                                    batch_i,
                                                                                    loss.item() / targets.size(0),
                                                                                    train_loss / total,
                                                                                    correct / total,
                                                                                    duration))


    if IS_ON_NSML:
        nsml.report(
            summary=True,
            step=epoch,
            epoch_total=epoch_times,
            train_loss=train_loss / total,
            train_acc=correct / total,
            lr=scheduler.get_last_lr()
        )

        nsml.save(str(epoch + 1))

    scheduler.step()
    print('Training   Epoch=%d, Loss=%.3f, Acc=%.3f' % (epoch, train_loss / total, correct / total))

def _infer(model, root_path):

    test_loader = test_data_loader(
        root=os.path.join(root_path, 'test_label'),
        phase='test',
        batch_size=64
    )

    res_fc = None
    res_id = None
    print(model)
    for idx, (data_id, image) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        fc = model(image)
        fc = fc.detach().cpu().numpy()

        if idx == 0:
            res_fc = fc
            res_id = data_id
        else:
            res_fc = np.concatenate((res_fc, fc), axis=0)
            res_id = res_id + data_id

    res_cls = np.argmax(res_fc, axis=1)


    return [res_id, res_cls]

def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return _infer(model, path)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.



if __name__ == '__main__':
    global device
    global val_loader
    global tr_loader
    global optimizer
    global criterion
    global vgg16_ft
    global epoch_times

    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--train_split", type=float, default=0.9)
    args.add_argument("--num_classes", type=int, default=6)
    args.add_argument("--lr", type=int, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--eval_split", type=str, default='val')
    args.add_argument("--batch_size", type=int, default=32)

    # reserved for nsml
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    train_split = config.train_split
    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    eval_split = config.eval_split
    batch_size = config.batch_size
    mode = config.mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ################ models #################
    # Densenet161
    densenet161_2d = models.densenet161(pretrained=True)
    densenet161_2d.features.conv0 = nn.Conv2d(9, 96,
                                              kernel_size=(7, 7),
                                              stride=(2, 2),
                                              padding=(3, 3),
                                              bias=False)
    densenet161_2d.classifier = nn.Sequential(nn.Linear(in_features=2208, out_features=256),
                                              nn.ReLU(),
                                              nn.Dropout(p=0.2),
                                              nn.Linear(256, 10))

    # Resnet152
    resnet_2d = models.resnet152(pretrained=True)
    resnet_2d.conv1 = nn.Conv2d(9, 64,
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=(3, 3),
                                bias=False)
    resnet_2d.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=256),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2,),
                                 nn.Linear(256, 10))

    # EfficientNet_b0
    efficient_net = models.efficientnet_b0(pretrained=True)
    efficient_net.features[0][0] = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    efficient_net.classifier[1] = nn.Sequential(nn.Linear(1280, 256),
                                                nn.Dropout(p=0.2, inplace=True),
                                                nn.Linear(256, 10))

    # (2+1)D resnet
    resnet_2p1d = models.video.r2plus1d_18(pretrained=False)
    resnet_2p1d.layer3[0].relu = nn.Sequential(nn.ReLU(inplace=True),
                                               nn.Flatten(start_dim=1, end_dim=2))

    resnet_2p1d.layer3[1].conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True))
    resnet_2p1d.layer3[1].conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True))

    resnet_2p1d.layer4[0].conv1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True))
    resnet_2p1d.layer4[0].conv2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
                                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True))
    resnet_2p1d.layer4[0].downsample = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                )

    resnet_2p1d.layer4[1].conv1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
                                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True))
    resnet_2p1d.layer4[1].conv2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
                                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True))
    resnet_2p1d.avgpool = nn.AdaptiveAvgPool2d(1)
    resnet_2p1d.fc = nn.Linear(512, 10)

    # conv - bn - relu - conv - bn - relu
    #print(resnet_2p1d)
    #torchsummary.summary(resnet_2p1d, (3, 3, 224, 224))
    ################ models #################
    #resnet_2p1d = r2plus1d_18()

    # optimizer
    optimizer = optim.SGD(resnet_2p1d.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-3)

    # criterion
    criterion = nn.CrossEntropyLoss()

    #model = densenet161_2d
    model = resnet_2p1d
    model = model.to(device)

    if IS_ON_NSML:
        bind_model(model, optimizer)

    if config.pause:
        nsml.paused(scope=locals())
        
    if config.mode =='train':
        # epoch times
        epoch_times = 1
        start_epoch = 0
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_times, eta_min=1e-5)

        best_acc = 0

        tr_loader, val_loader, val_label = data_loader_with_split(
                root=TRAIN_DATASET_PATH,
                train_split=train_split,
                batch_size = batch_size,
                data_mode='2d'
            )


        for epoch in range(start_epoch, start_epoch + epoch_times):
            train(epoch,model,scheduler)
            validate(epoch, model)
