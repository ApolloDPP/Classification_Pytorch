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
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class RowUltra_data(Dataset):
    def __init__(self,dataroot,transforms=None):  #datapath为不同类图片的父文件路径:train/
        dirs=os.listdir(dataroot)  #存储每张图片路径
        self.transforms = transforms
        self.img_all =[]
        for dir in dirs:
            root1=os.path.join(dataroot,dir)
            self.imgs=[os.path.join(root1,imgpath) for imgpath in os.listdir(root1)]
            self.img_all=np.append(self.img_all,self.imgs )
        print(self.img_all)

    def __getitem__(self, item):
        img_path=self.img_all[item]
        data=Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data,img_path

    def __len__(self):
        return len(self.img_all )


def classifier(**kwargs):
    opt._parse(kwargs)
    # 导入模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    # model.to(opt.device)

    # 导入数据
    transform = T.Compose([
        T.Resize(300),  # 将最短resize至400，长宽比不变
        T.CenterCrop(227),  # 将中间大小400*400裁剪
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
    ])
    class_data = RowUltra_data(opt.chass_rawdata_root, transforms=transform)
    class_dataloader = DataLoader(class_data, batch_size=1,
                                 shuffle=False , num_workers=0)


    num=0
    for ii, (data, imgpath) in tqdm(enumerate(class_dataloader)):
        # input = data.to(opt.device)
        # target = label.to(opt.device)
        input = data
        score = model(input)
        score = score.detach().numpy()

        score = 0 if score[0][0] > score[0][1] else 1

        if score ==1:
            img=cv2.imread(imgpath[0])
            imgsave_path=os.path.join(opt.head_path +str(num) + "_" + "head.jpg")
            cv2.imwrite(imgsave_path, img)
            num=num+1




if __name__=='__main__':
    classifier()

