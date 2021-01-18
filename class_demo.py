#coding:utf8
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
import cv2



def classifier(**kwargs):
    opt._parse(kwargs)
    # 导入模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    # model.to(opt.device)

    # 导入数据
    transform = T.Compose([
        T.Resize(600),  # 将最短resize至400，长宽比不变
        T.CenterCrop(400),  # 将中间大小400*400裁剪
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
    ])
    class_data = Ultrasound_data(opt.chass_rawdata_path, transforms=transform)
    class_dataloader = DataLoader(class_data, batch_size=1,
                                 shuffle=False , num_workers=0)


    for ii, (data, label,imgpath) in tqdm(enumerate(class_dataloader)):
        # input = data.to(opt.device)
        # target = label.to(opt.device)
        input = data
        score = model(input)
        score = score.detach().numpy()
        target = label

        score = 0 if score[0][0] > score[0][1] else 1

        if score ==1:
            img=cv2.imread(imgpath[0])
            imgsave_path=os.path.join(opt.head_path +str(ii) + "_" + "head.jpg")
            cv2.imwrite(imgsave_path, img)




if __name__=='__main__':
    classifier()

