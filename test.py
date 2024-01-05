import openslide
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import pylab
import os
import torch
from tqdm import tqdm, trange
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from PIL import Image
# import moxing as mox
import pandas as pd
import cv2
# import moxing as mox

#Parameter Definition
def get_data_from_obs():
    #癌症切片
    data = mox.file.list_directory('obs://wzclinai1/', recursive=True)
    for name in tqdm(data):
        mox.file.copy('obs://wzclinai1/' + name,'/home/ma-user/work/data/cancer/' + name)

    #正常切片
    # name = '2023-11-01 19_11_18.svs'
    # mox.file.copy('obs://migrationbackup/zcqp/' + name,'/home/ma-user/work/data/cancer_free/' + name)
    data = mox.file.list_directory('obs://migrationbackup/zcqp/', recursive=True)
    for name in tqdm(data):
        try:
            mox.file.copy('obs://migrationbackup/zcqp/' + name,'/home/ma-user/work/data/cancer_free/' + name)
        except:
            continue
    
    print(data)
    with open('../../obs_data.txt') as f:
        data = f.read()
        data_list = data.split('<Key>')
        print(data_list)

    mox.file.copy('obs://wzclinai1','/home/ma-user/work/data/wzclinai1')
    filelist = os.listdir('obs://wzclinai1/')
    for file in filelist:
        print(file)

def collect_dataset():
    '''
    image.level_count: View the number of image level, level refers to the hierarchy of the image
    image.level_downsamples: Downscaling a level of a WSI
    '''
    slides = []
    grid = []
    targets = []
    mult = 1
    level = 0

    number_data = []
    #Process prepare the train and test data
    filedir = ('data')
    filelist = os.listdir(filedir)
    
    for file in filelist:
        subfilelist = os.listdir(filedir + '/' +file)
        data_count = 0
        for i, subfile in tqdm(enumerate(subfilelist)):
            # print(subfile)
            image = openslide.OpenSlide(filedir + '/' + file + '/' + subfile)
            if image.level_count < 3:
                continue
            # if image.level_count != 7:
            #     continue
            # print(os.path.abspath(subfile).replace('\\', '/')) 
            slides.append(filedir + '/' + file + '/' + subfile)
            # print(image.level_downsamples[0])
            # print(image.get_best_level_for_downsample(98932))
            subgrid = []
            # print(image.level_count)
            # for j in range(image.level_count):
            #     subgrid.append(image.level_dimensions[j])

            #最低级别为3，因此全部设定为3
            for j in range(3):
                subgrid.append(image.level_dimensions[j])
                # print(image.level_dimensions[j])
            grid.append(subgrid)
            # print(file.split('_')[1])
            targets.append(int(file.split('_')[-1]))
            data_count += 1
        number_data.append(data_count)

    return slides, grid, targets, mult, level, number_data

def set_train_val_test_dataset(train_split, val_split):
    slides, grid, targets, mult, level, number_data = collect_dataset()
    test_split = 1 - train_split - val_split
    train_data_split = list(map(lambda x:int(x*train_split),number_data))
    val_data_split = list(map(lambda x:int(x*val_split),number_data))
    test_data_split = list(map(lambda x:int(x*test_split),number_data))

    database = number_data[0]
    tr_0 = train_data_split[0]
    tr_1 = train_data_split[1]

    train_data = {
        'slides': slides[:tr_0] + slides[database:database+tr_1],
        'grid': grid[:tr_0] + grid[database:database+tr_1],
        'targets': targets[:tr_0] + targets[database:database+tr_1], 
        'mult': mult,
        'level': level,
        }
    print('train_data : ', len(train_data['slides']))
    print(train_data['targets'])
    
    va_0 = val_data_split[0]
    va_1 = val_data_split[1]

    val_data = {
        'slides': slides[tr_0:tr_0+va_0] + slides[database+tr_1:database+tr_1+va_1],
        'grid': grid[tr_0:tr_0+va_0] + grid[database+tr_1:database+tr_1+va_1],
        'targets': targets[tr_0:tr_0+va_0] + targets[database+tr_1:database+tr_1+va_1], 
        'mult': mult,
        'level': level,
        }
    print('val_data : ', len(val_data['slides']))
    print(val_data['targets'])

    test_data = {
        'slides': slides[tr_0+va_0:database] + slides[database+tr_1+va_1:],
        'grid': grid[tr_0+va_0:database] + grid[database+tr_1+va_1:],
        'targets': targets[tr_0+va_0:database] + targets[database+tr_1+va_1:], 
        'mult': mult,
        'level': level,
    }
    print('test_data : ', len(test_data['slides']))
    print(test_data['targets'])
    
    #所有图像全部输入
    # if not os.path.exists('origin_data/'):
    #     os.mkdir('origin_data/')
    # torch.save(train_data, 'origin_data/train_data')
    # torch.save(val_data, 'origin_data/val_data')
    # torch.save(test_data, 'origin_data/test_data')

    #仅使用级别为7的图像
    # if not os.path.exists('7cls_data/'):
    #     os.mkdir('7cls_data/')
    # torch.save(train_data, '7cls_data/train_data_7cls')
    # torch.save(val_data, '7cls_data/val_data_7cls')
    # torch.save(test_data, '7cls_data/test_data_7cls')

    #仅使用图像的3个级别
    if not os.path.exists('3cls_data/'):
        os.mkdir('3cls_data/')
    torch.save(train_data, '3cls_data/train_data_3cls')
    torch.save(val_data, '3cls_data/val_data_3cls')
    torch.save(test_data, '3cls_data/test_data_3cls')

    print('Finish saving train_data val_data and test_data!!!')

def open_image_svs():
    path = 'cancer-free_0/2023-11-21 15_23_38.svs'
    image = openslide.OpenSlide('data/' + path)
    print(image.level_count)
    for i in range(image.level_count):
        print(image.level_dimensions[i])
        
    #(0,0）代表起始像素点，2代表需要处理的层级，（224,224)代表处理得到的像素大小
    # result = np.array(image.read_region((0, 0), 3, (224, 224)))
    if not os.path.exists('WSI_show/' + path.split('.')[0]):
            os.mkdir('WSI_show/' + path.split('.')[0])
    for i in range(3):
        result = np.array(image.read_region((0, 0), i+3, image.level_dimensions[i+3]).convert('RGB'))
        print(result.shape)
        plt.figure(i)
        plt.imshow(result)
        # plt.savefig('WSI_show/' + path.split('.')[0] + '/level_' + str(i+3) +'.png')
        plt.show()

def count_number():
    print()

def main():
    image = openslide.OpenSlide('data/cancer_1/2023-11-21 12_18_01.svs')
    # print(image)
    # data = torch.load('train_data')
    # print(data)
    # print(type(data))
    # slides_train = torch.load('train_data')
    # print(slides_train)
    # train_data = pd.DataFrame(slides_train['slides'])
    # torch.save(slides['slides'], 'train_data.txt')
    # slides_val = torch.load('val_data')
    # print(slides_val)
    # val_data = pd.DataFrame(slides_val['slides'])
    # train_data.to_csv('train_data.csv')
    # val_data.to_csv('val_data.csv')
    # torch.save(slides['slides'], 'val_data.txt')
    # pred = [1,1,1,1]
    # real = [1,1,1,1]
    # pred = np.array(pred)
    # real = np.array(real)
    # neq = np.not_equal(pred, real)
    # err = float(neq.sum())/pred.shape[0]
    # print(pred)
    # print(real)
    # print(neq.sum())
    # fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    # print(np.logical_and(pred==1,neq))

    # fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()

class DoubleAttention(nn.Module):

    def __init__(self, in_channels,c_m,c_n,reconstruct = True):
        super().__init__()
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv2d(in_channels,c_m,1)
        self.convB=nn.Conv2d(in_channels,c_n,1)
        self.convV=nn.Conv2d(in_channels,c_n,1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,h,w
        B=self.convB(x) #b,c_n,h,w
        V=self.convV(x) #b,c_n,h,w
        tmpA=A.view(b,self.c_m,-1)
        print('tmpA', tmpA.shape)
        attention_maps=F.softmax(B.view(b,self.c_n,-1))
        attention_vectors=F.softmax(V.view(b,self.c_n,-1))
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ=tmpZ.view(b,self.c_m,h,w) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)

        return tmpZ 





if __name__ == '__main__':
    # collect_dataset()
    # set_train_val_test_dataset(0.6, 0.2)
    # set_test_dataset()
    # get_data_from_obs()
    # open_image_svs()
    # main()
    input=torch.randn(50,512,7,7)
    a2 = DoubleAttention(512,128,128,True)
    print(a2)
    output=a2(input)
    print(output.shape)
