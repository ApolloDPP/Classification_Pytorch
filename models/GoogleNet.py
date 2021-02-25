import torch
from torch import nn

class Inception(nn.Module):
    #in_c为模块的输出通道数，c1-c4是每条通路上的输出通道数，第2，3路的c为2个参数
    def __init__(self,in_c,c1,c2,c3,c4):
        super(Inception,self).__init__()
        
        self.path1=nn.Sequential(
            nn.Conv2d(in_c,c1,kernel_size=1),
            nn.ReLU()
        )
        self.path2=nn.Sequential(
            nn.Conv2d(in_c,c2[0],kernel_size=1),
            nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.path3=nn.Sequential(
            nn.Conv2d(in_c,c3[0],kernel_size=1),
            nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2),
            nn.ReLU()
        )
        self.path4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=2),
            nn.Conv2d(in_c,c4,kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self,x):
        return torch.cat(self.path1(x),self.path2(x),self.path3(x),self.path4(x))
#net=Inception(3,10,[10,20],[20,10],30)

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        
        self.block1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,padding_mode=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block2=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64,192,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block3=nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block4=nn.Sequential(
            Inception(480,192,(96,208),(16,48),64),
            Inception(512,160,(112,224),(24,64),64),
            Inception(512,128,(128,256),(24,64),64),
            Inception(512,112,(144,288),(32,64),64),
            Inception(528,256,(160,320),(32,128),128),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block5=nn.Sequential(
            Inception(832,256,(160,320),(32,128),128),
            Inception(832,384,(192,384),(48,128),128),
            nn.AvgPool2d(kernel_size=7),
            nn.Linear(1024,10)
        )
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        return x
net=GoogleNet()
