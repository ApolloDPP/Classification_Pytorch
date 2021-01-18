#coding:utf8

import torch

class Default_Config(object):
    env="default"
    vis_port=8097
    model="Resnet34"

    train_data_root="D:\\SIGS\\ultradata_classifier\\data\\train"
    #test_data_root="./data/test"
    load_model_path="D:\\SIGS\\ultradata_classifier\\checkpoints\\resnet34_10_42.pth "
    chass_rawdata_root="D:\\SIGS\\ultradata_Dong\\Part1"
    head_path = "D:\\SIGS\\ultradata_classifier\\head\\"

    batch_size=4
    use_gpu=False
    num_workers=0

    max_epoch=100
    lr=0.001
    lr_decay=0.5
    weight_decay = 0e-5

    print_freq=20

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt=Default_Config()