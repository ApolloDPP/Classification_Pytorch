# coding:utf8
"""
封装常见的model功能：save,load,optimizer
"""
import torch
from torch.nn import Module
import time

class Basic_model(Module ):
    def __init__(self):
        super(Basic_model, self).__init__()
        self.model_name = str(type(self))

    def load(self,path):
        self.load_state_dict(torch.load(path))

    def save(self,name=None):
        if name is None:
            prefix="checkpoints/"+self.model_name+"_"
            name=time.strftime(prefix + '%m%d_%H:%M.pth')
        torch.save(self.state_dict(),name)
    def get_optimizer(self,lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


