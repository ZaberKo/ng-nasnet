from torchsummaryX import summary
import json
from nasnet import NASNetCIFAR
import torch


def main():
    cell_config_list = {'normal_cell': normal_cell_config}
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
        steps=100,
    )
    # model.eval()
    summary(model, torch.zeros((1, 3, 32, 32)))


if __name__ == '__main__':

    config_path = 'config.json'

    with open(config_path, mode='r', encoding='utf-8') as f:
        config = json.load(f)

    train_config = config['train_config']
    nasnet_config = config['nasnet_config']

    # top1
    normal_cell_config = {
        2: [(0, 1), (1, 7)],
        3: [(0, 3), (1, 1),(2, 1)],
        4: [(0, 1), (1, 1), (2, 6)],
        5: [(0, 7), (1, 1), (4, 7)],
        6: [(0, 1), (1, 7)],
        7: [(3, 1), (5, 1), (6, 1)]
    }

    # top2
    normal_cell_config = {
        2: [(0, 1), (1, 7)],
        3: [(0, 4), (1, 1), (2, 6)],
        4: [(0, 7), (1, 3)],
        5: [(0, 5), (1, 7), (4, 6)],
        6: [(0, 7), (1, 1), (2, 1), (3, 7)],
        7: [(5, 1), (6, 1)]
    }

    # top3
    normal_cell_config = {
        2: [(0, 1)],
        3: [(0, 3), (1, 1)],
        4: [(1, 6), (2, 4)],
        5: [(0, 5), (1, 1), (2, 7), (4, 7)],
        6: [(0, 7),  (3, 6), ],
        7: [(5, 1), (6, 1)]
    }

    # top4
    normal_cell_config = {
        2: [(0, 1)],
        3: [(0, 5), (1, 5), (2, 1)],
        4: [(0, 4), (1, 6)],
        5: [(0, 3), (1, 7)],
        6: [(0, 7), (1, 1), (2, 2)],
        7: [(3, 1), (4, 1), (5, 1), (6, 1)]
    }

    # # nasnet-A
    # normal_cell_config = {
    #     2: [(1, 7), (1, 7)],
    #     3: [(0, 7), (1, 6)],
    #     4: [(0, 1), (1, 2)],
    #     5: [(0, 2), (1, 2)],
    #     6: [(0, 7), (1, 7)],
    #     7: [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
    # }

    main()
