from torch import nn

def vgg_block(num_convs,in_channels,out_channels):
    block=[]  #存储模块结构的list
    for i in range(num_convs):
        if i==0:  #每个模块只有两个channels不同，先传入输入channels
            block.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        else:
            block.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        block.append(nn.ReLU())
    block.append(nn.MaxPool2d(2,2))   #每个卷积模块之后池化，图大小减半
    return nn.Sequential(*block) #通配符的使用

def vgg(conv_arch):
    net=nn.Sequential()
    for i,(num_convs,in_channels,out_channels) in enumerate(conv_arch):
        net.add_module("block"+str(i),vgg_block(num_convs,in_channels,out_channels))
    
    net.add_module("fc",nn.Sequential(
        nn.Linear(512*7*7,4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,10)
    ))
    return net

conv_arch=((2,3,64),(2,64,128),(3,128,256),(3,256,512),(3,256,512))
vgg16=vgg(conv_arch)
