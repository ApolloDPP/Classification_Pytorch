from torch import nn
import torch.nn.functional as F

class NinBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(NinBlock,self).__init__()
        
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU()          
        )
    def forward(self,x):
        output=self.block(x)
        return output
#########################################    
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d,self).__init__()
    def forward(self,x):
        return F.avg_pool2d(x,kernel_size=x.size()[2:])
############################## 
NinNet=nn.Sequential(
    NinBlock(1,96,kernel_size=11,stride=4,padding=0),
    nn.MaxPool2d(kernel_size=3,stride=2),
    NinBlock(96,256,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=3,stride=2),
    NinBlock(256,384,kernel_size=3,stride=1,padding=1),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Dropout(0.5),
    
    NinBlock(384,10,kernel_size=3,stride=1,padding=1),
    GlobalAvgPool2d()
    
)
print(NinNet)
