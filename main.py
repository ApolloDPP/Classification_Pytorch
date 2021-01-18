from config import opt
import os
import torch as t
import models
from data.dataset import Ultrasound_data
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import visualizer
from tqdm import tqdm
from torchvision  import transforms as T

def train(**kwargs ):
    opt._parse(kwargs)
    #vis = visualizer(opt.env, port=opt.vis_port)

    #配置模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    #导入数据
    transform = T.Compose([
        T.Resize(400),  # 将最短resize至400，长宽比不变
        T.CenterCrop(400),  # 将中间大小400*400裁剪
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
    ])
    train_data = Ultrasound_data (opt.train_data_root,transforms=transform )
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    # 损失函数与优化
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # 计算误差
    loss_meter = meter.AverageValueMeter()
    #confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    # train
    for epoch in range(opt.max_epoch):
        loss_epoch=0

        loss_meter.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            # train model
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            loss_epoch=loss_epoch+loss.detach().numpy()
        print("epoch=",epoch,"Loss=",loss_epoch/600)

        model.save()

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]


def test(**kwargs):
    opt._parse(kwargs)
    #导入模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    #model.to(opt.device)

    #导入数据
    transform = T.Compose([
        T.Resize(400),  # 将最短resize至400，长宽比不变
        T.CenterCrop(400),  # 将中间大小400*400裁剪
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
    ])
    test_data = Ultrasound_data(opt.train_data_root, transforms=transform)
    test_dataloader = DataLoader(test_data, batch_size=1,
                                  shuffle=True, num_workers=0)

    #test
    """
    混淆矩阵：
        truth   0    1
    pre
    
    0           a     b 
    
    1           c     d
    """
    a=0
    b=0
    c=0
    d=0
    for ii, (data, label) in tqdm(enumerate(test_dataloader)):
        #input = data.to(opt.device)
        #target = label.to(opt.device)
        input = data
        score = model(input)
        score =score .detach().numpy()
        target = label

        score =0 if score [0][0]<0 else 1


        if score ==0 and target ==0:
            a=a+1
        elif score ==1 and target ==1:
            d=d+1
        elif score ==1 and target ==0:
            c=c+1
        else :
            d=d+1
    print("a=",a)
    print("b=", b)
    print("c=", c)
    print("d=", d)


if __name__=='__main__':
    test()