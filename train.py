import time

import torch.nn as nn
import torch
import torch.optim
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import numpy as np
from utils import update_dropout_schedule

from nasnet import NASNetCIFAR

import argparse
import json

from apex import amp


def train(model, trainloader, testloader, optimizer, device):
    loss_func = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    total_step = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(train_config['epoch']):
        start_time = time.time()
        model.train()

        for step, data in enumerate(trainloader, 0):
            data = tuple(t.to(device) for t in data)
            images, labels = data
            output = model(images)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            if train_config['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_step += 1
            if step % 10 == 9:
                print('epoch: {0}, iter:{1} loss:{2:.4f} avg_loss:{3:.4f}'.format(
                    epoch, step, loss.item(), running_loss / total_step))

        print('epoch{} test: '.format(epoch), end="")
        evaluate(model, testloader, device)
        # print('epoch{} train: '.format(epoch), end="")
        # evaluate(model, trainloader)
        print('epoch {} finished, cost {:.3f} sec'.format(
            epoch, time.time() - start_time))
        print('=======================\n\n\n')

        update_dropout_schedule(model)


def evaluate(model: torch.nn.Module, testloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            data = tuple(t.to(device) for t in data)
            images, labels = data

            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # if total>100:
            #     break

    acc = correct / total * 100
    print(
        'Accuracy of the network on the 10000 test images: {:.3f}'.format(acc))

    return acc


def load_dataset(path: str, batch_size: int):
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    trainset = datasets.CIFAR10(root=path, train=True,
                                download=True, transform=transforms.Compose(transf+normalize))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2,pin_memory=True)

    testset = datasets.CIFAR10(root=path, train=False,
                               download=True, transform=transforms.Compose(normalize))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2,pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def save_model():
    pass


def load_model():
    pass


def main():
    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    batch_size = train_config['batch_size']

    trainloader, testloader, classes = load_dataset(
        train_config['data_path'], batch_size)

    cell_config_list = {'normal_cell': normal_cell_config}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    steps = train_config['epoch']

    model = NASNetCIFAR(
        cell_config=cell_config_list,
        stem_channels=nasnet_config['num_stem_channels'],
        cell_base_channels=nasnet_config['cell_base_channels'],
        num_stack_cells=nasnet_config['num_stack_cells'],
        image_size=32,
        num_classes=10,
        start_dropblock_prob=train_config['start_dropblock_rate'],
        end_dropblock_prob=train_config['end_dropblock_rate'],
        start_droppath_prob=train_config['start_droppath_rate'],
        end_droppath_prob=train_config['end_droppath_rate'],
        dropfc_prob=train_config['dropfc_rate'],
        steps=steps,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'],weight_decay=0.0001)

    model.to(device)
    if train_config['fp16']:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if n_gpu > 1:
        model = nn.DataParallel(model)

    train(model, trainloader, testloader, optimizer, device)

    # evaluate(model, testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config.json',
                        type=str, help="config file path")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, mode='r', encoding='utf-8') as f:
        config = json.load(f)

    train_config = config['train_config']
    nasnet_config = config['nasnet_config']
    normal_cell_config = {
        2: [(0, 1), (1, 7)],
        3: [(0, 4), (2, 7)],
        4: [(2, 1), (3, 4)],
        5: [(2, 1), (3, 7), (4, 7)],
        6: [(0, 7), (3, 6), (4, 1)],
        7: [(5, 1), (6, 1)]
    }

    # normal_cell_config = {
    #     2: [(0, 1), (1, 7)],
    #     3: [(0, 4), (2, 7)],
    #     4: [(2, 1), (3, 4)],
    #     5: [(2, 1), (3, 7), (4, 7)],
    #     6: [(0, 7), (3, 6), (4, 1)],
    #     7: [(5, 1), (6, 1)]
    # }
    main()
