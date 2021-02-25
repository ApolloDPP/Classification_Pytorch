import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion=1  #为了计算Basic/Bottle两个模块的输出通道设定的常数
    #BasicBlock是Resnet18/34的子模块
    def __init__(self,in_channels,out_channels,stride=1):
        super(BasicBlock,self).__init__()
        
        self.residual=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels*BasicBlock.expansion,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
        )
        
        self.shorcut=nn.Sequential(
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
        )
        
        #在x与f(x)不同维度时需要1x1卷积match
        if in_channels!=out_channels*BasicBlock.expansion or stride!=1:
            self.shorcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels*BasicBlock.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion)
            )
            
    def forward(self,x):
        return nn.ReLU(inplace=True)(self.residual(x)+slef.shorcut(x))

##################################################
class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_channels,out_channels,stride=1):
        super(Bottleneck,self).__init__()
        self.residual=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,stride=stride,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels*Bottleneck.expansion,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels*Bottleneck.expansion)        
        )
        
        self.shorcut=nn.Sequential(
            nn.BatchNorm2d(out_channels*Bottleneck.expansion)
        )
        
        #在x与f(x)不同维度时需要1x1卷积match
        if in_channels!=out_channels*Bottleneck.expansion or stride!=1:
            self.shorcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels*Bottleneck.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*Bottleneck.expansion)
            )
            
    def forward(self,x):
        return nn.ReLU(inplace=True)(self.residual(x)+slef.shorcut(x))

class RestNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super(RestNet,self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        #每个模块里面的stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18():
    return RestNet(BasicBlock,[2,2,2,2])

def resnet34():
    return RestNet(BasicBlock,[3,4,6,3])

def resnet50():
    return RestNet(Bottleneck,[3,4,6,3])

def resnet101():
    return RestNet(Bottleneck,[3,4,23,3])

def resnet152():
    return RestNet(Bottleneck,[3,8,36,3])
