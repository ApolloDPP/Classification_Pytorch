"""
将后缀相同的图片进行名称转换
将不同类的图片放在统一文件夹，结构如下：
data/:
    head    里面是头部的图片，后缀一致即可
    femoral
    abandom
"""
import os
import cv2

# train的绝对路径，train下面包含三个文件夹：abdomen,head,feomral,每个类别的图片分别放置在相应的文件夹内
filepath = "D:\\SIGS\\ultra_1_18\\"
#print(folderlist )
def namechange(filepath):
    folderlist = os.listdir(filepath)# 列举文件夹：也即abdomen,head,feomral三个文件夹
    for folder in folderlist:
        inner_path = os.path.join(filepath , folder)
        print(inner_path)
        total_num_folder = len(folderlist)  # 文件夹的总数
        print('total have %d folders' % (total_num_folder))  # 打印文件夹的总数

        filelist = os.listdir(inner_path)  # 列举每个文件夹下面的图片
        print("图片的总数:",len(filelist))
        i = 0
        for item in filelist:
            total_num_file = len(filelist)  # 单个文件夹内图片的总数
            if item.endswith('.jpg'):    #原图的后缀
                src = os.path.join(os.path.abspath(inner_path), item)  # 原图的地址
                dst = os.path.join(os.path.abspath(inner_path), str(i) + '_'+ str(folder) + '.png')
                # 新图的地址（这里可以把str(folder) + '_' + str(i) + '.jpg'改成你想改的名称）
                #png是新图的后缀
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i += 1
                except:
                    continue

path1 = "D:\\SIGS\\ultradata_classifier\\data\\change\\head\\"  # 第一步所生成的的三个文件夹（每个类别）
path2 = "D:\\SIGS\\ultradata_classifier\\data\\change\\other\\"  # 最终需要的train/test文件目录（新建）
path3 = "D:\\SIGS\\ultradata_classifier\\data\\train\\"
def datacut(path1,path2):
    list = os.listdir(path1)
    L = len(list)  # 每个类别图片的数量
    for i in range(L):  # 将每张图片读取并开始预处理
        img_path = os.path.join(path1, str(i) + "_" + "other.png")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (700, 700))
        imgcut = img[140:560, 150:570]  # 裁剪（本实验在shape=[575,767]大小的图片中只截取了中间420X420大小的部分）
        print(imgcut.shape)
        imgcut_path = os.path.join(path2, str(i + 300) + "_" + "other.png")
        # 特别注意：str(i+166)中，166的作用就是将第二类在第一类的基础上继续排号，这儿的head有155张，则femoral从166开始排序
        cv2.imwrite(imgcut_path, imgcut)  # 把预处理完的图片保存到test文件夹中

if __name__ == '__main__':
    namechange(filepath)
    #datacut(path2, path3)