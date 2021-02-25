from torch import nn

#稠密块DenseBlock
class DenseBlock(nn.Module):
    def __init__(self,num_conv,in_channels,out_channels):
        super(DenseBlock,self).__init__()
        net=[]
        for i in range(num_conv):
            in_channels=in_channels+i*out_channels
            net.append(self._conv_block(in_channels,out_channels))
        self.net=nn.ModuleList(net)
        #连结之后的最终输出通道数out_channels
        self.out_channels=in_channels+num_conv*out_channels
        
        #将卷积三件套封装起来
    def _conv_block(self,in_channels,out_channels):
        conv_block=nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        )
        return conv_block
    
    def forward(self,x):
        #每层卷积之后将输出和输入连结之后送入下一层卷积
        for block in self.net:
            y=block(x)
            x=torch.cat((x,y),dim=1)
            return x
        
#过渡层TransitionLayer
def transitionLayer(in_channels,out_channels):
    net=nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels,out_channels,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )
    return net
########################################
class DenseNet(nn.Module):
    #in_channels:稠密层开始的输入，和resnet一样先是经过7x7，maxpooling操作
    #growth_rate:每层conv之后增加的通道数（也即DenseBlock中的out_channels参数）
    #num_conv:每个稠密块包含的卷积层数
    def __init__(self,in_channels,growth_rate,num_convs):
        super(DenseNet,self).__init__()
        #in_channels,growth_rate=64,32
        #num_convs=[4,4,4,4]
        self.net=nn.Sequential()
        
        conv1=nn.Sequential(
             nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
         )
        
        self.net.add_module("Block1",conv1)
        
        for i ,num_conv in enumerate(num_convs):
            block=DenseBlock(num_conv,in_channels,growth_rate)
            self.net.add_module("Block%d"%(i+2),block)
            in_channels=block.out_channels
            
            #在最后稠密块之后加入过渡层
            if i !=len(num_convs)-1:
                self.net.add_module("TransitionLayer% d"%i,transitionLayer(in_channels,in_channels//2))
                in_channels=in_channels//2
        
        #最后，与Resnet一样接入FC和Avgpool层
        def forward(self,x):
            x=self.net(x)
            output = self.avg_pool(x)
            output = output.view(output.size(0), -1)
            output = self.fc(output,10)
            return output

in_channels,growth_rate=64,32
num_convs=[4,4,4,4]
net=DenseNet(in_channels,growth_rate,num_convs)
print(net)
