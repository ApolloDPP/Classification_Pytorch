from torch import nn

class LeNet(nn.Module): #一般都继承nn.Moudle，里面包含常用的模型属性
    def __init__(self):
        super (LeNet,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,6,5), #in-channels,out-channels,kernel-size
            nn.Sigmoid(),
            nn.MaxPool2d(2,2), #kernel-size,stride
            
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)         
        )
        
        self.fc=nn.Sequential(
            nn.Linear(256,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )
        
    def forward(self,img):
        feature=self.conv(img)
        feature=feature.view(img.shape[0],-1)#将特征图flatten化，喂入linear层
        output=self.fc(feature)
        return output
net=LeNet()
