import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision  import transforms as T
#from torch.utils import data

transform=T.Compose ([
    T.Resize (400),#将最短resize至400，长宽比不变
    T.CenterCrop (400),#将中间大小400*400裁剪
    T.ToTensor (),
    T.Normalize (mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
    ])

class Ultrasound_data(Dataset):
    def __init__(self,datapath,transforms=None):  #datapath为不同类图片的父文件路径:train/
        imgs=os.listdir(datapath)  #存储每张图片路径
        self.imgs=[os.path.join(datapath,img) for img in imgs]
        self.transforms=transforms
    def __getitem__(self, item):
        img_path=self.imgs[item]
        print(img_path )
        label=1 if "head" in img_path.split("/")[-1] else 0
        print(label)
        data=Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data,label,img_path
    def __len__(self):
        return len(self.imgs )

"""
datapath="D:\\SIGS\\ultradata_classifier\\data\\train"
dataset=Ultrasound_data (datapath ,transforms=transform)
#img,label=dataset [0]
for img,label in dataset:
    print(img.size(),label)
"""


