import torch
from torch import nn



def get_stack_out_channels(cellstack):
    cells=cellstack.cells
    assert len(cells) >= 1
    out_channels = cells[-1].out_channels
    if len(cells) == 1:
        out_prev_channels = cells[-1].in_channels
    else:
        out_prev_channels = cells[-2].out_channels

    return out_channels,out_prev_channels





class LinearScheduler(nn.Module):
    def __init__(self, module, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.module = module
        self.iter = 0
        self.drop_values = torch.linspace(start=start_value, end=stop_value, steps=nr_steps)


    def forward(self, x):
        return self.module(x)

    def step(self):
        if self.iter < len(self.drop_values):
            self.module.drop_prob = self.drop_values[self.iter]

        self.iter += 1


def _check_and_update(m):
    if isinstance(m,LinearScheduler):
        m.step()


def update_dropout_schedule(module:nn.Module):    
    module.apply(_check_and_update)
        
        
