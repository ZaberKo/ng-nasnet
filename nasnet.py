import torch

import torch.nn as nn
from collections import OrderedDict
from cell_elem import *
from utils import *
from cell import *
from dropblock import ScheduleDropBlock
from droppath import ScheduleDropPath_
'''
    cell config description
    0,1: input node
    2~n-2: hidden state node
    n-1: output node
'''


class NASNetCIFAR(nn.Module):
    def __init__(self, 
    cell_config: dict, 
    stem_channels:int,
    cell_base_channels: int, 
    num_stack_cells: int, 
    image_size: int, 
    num_classes: int,
    start_dropblock_prob:float,
    end_dropblock_prob:float,
    start_droppath_prob:float,
    end_droppath_prob:float,
    dropfc_prob:float,
    steps:int):
        super(NASNetCIFAR, self).__init__()
        '''
            steps: the number of mini-batch iterations
        '''
        self.num_classes = num_classes
        normal_cell_config = cell_config['normal_cell']
        reduce_cell_config=None

        _image_size = image_size
        _channels=cell_base_channels

        self.scheduled_droppath=ScheduleDropPath_(start_droppath_prob,end_droppath_prob,steps)

        self.cellstem=CellStem0(3,stem_channels)
        self.normal_layer0=FirstCellStack(
            config=normal_cell_config,
            image_size=_image_size,
            num_stack_cells=num_stack_cells,
            in_channels=stem_channels,
            conv_channels=_channels,
            scheduled_droppath=self.scheduled_droppath
        )
        self.reduction_layer0 = ReduceCell()
        self.dropblock0=ScheduleDropBlock(7,start_dropblock_prob,end_dropblock_prob,steps)
        
        _image_size//=2
        _channels*=2
        out_channels0,_=get_stack_out_channels(self.normal_layer0)
        self.normal_layer1 = NormalCellStack(
            config=normal_cell_config,
            image_size=_image_size,
            num_stack_cells=num_stack_cells,
            in_channels=out_channels0,
            in_prev_channels=out_channels0,
            conv_channels=_channels,
            scheduled_droppath=self.scheduled_droppath
            )
        self.reduction_layer1 = ReduceCell()
        self.dropblock1=ScheduleDropBlock(5,start_dropblock_prob,end_dropblock_prob,steps)
        
        _image_size //= 2
        _channels*=2
        out_channels1,_=get_stack_out_channels(self.normal_layer1)
        self.normal_layer2 = NormalCellStack(
            config=normal_cell_config,
            image_size=_image_size,
            num_stack_cells=num_stack_cells,
            in_channels=out_channels1,
            in_prev_channels=out_channels1,
            conv_channels=_channels,
            scheduled_droppath=self.scheduled_droppath
            )
        self.reduction_layer2 = ReduceCell()
        _image_size //= 2

        out_channels2,_=get_stack_out_channels(self.normal_layer2)
        self.gap_layer = nn.AdaptiveAvgPool2d(1)

        # self.softmax_layer = nn.Sequential(OrderedDict([
        #         ('fc1',nn.Linear(in_features=out_channels2, out_features=100)),
        #         ('fc2',nn.Linear(in_features=100, out_features=num_classes))
        #     ]))

        self.softmax_layer = nn.Sequential(OrderedDict([
                ('dropout1',nn.Dropout(dropfc_prob)),
                ('fc1',nn.Linear(in_features=out_channels2, out_features=100)),
                ('dropout2',nn.Dropout(dropfc_prob)),
                ('fc2',nn.Linear(in_features=100, out_features=num_classes))
            ]))

    

    def forward(self, images):
        x = self.cellstem(images)

        x,x_p=self.normal_layer0(x)
        x_new=self.reduction_layer0(x,x_p)
        x,x_p=x_new,x

        x=self.dropblock0(x)

        x,x_p=self.normal_layer1(x,x_p)
        x_new=self.reduction_layer1(x,x_p)
        x,x_p=x_new,x

        x=self.dropblock1(x)

        x,x_p=self.normal_layer2(x,x_p)

        x=self.gap_layer(x)

        x = x.view(x.size(0), -1)

        logits=self.softmax_layer(x)

        return logits


