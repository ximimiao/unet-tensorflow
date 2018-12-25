import numpy as np
import cv2
from scipy import misc


def color_annotation(img_path, output_path):
    '''
    给class图上色
    '''
    img = misc.imread(img_path)
    color = np.ones([img.shape[0], img.shape[1], 3])

    color[img==0] = [255, 255, 255] #其他，白色，0
    color[img==1] = [0, 255, 0]     #植被，绿色，1
    color[img==2] = [0, 0, 0]       #道路，黑色，2
    color[img==3] = [131, 139, 139] #建筑，黄色，3
    color[img==4] = [139, 69, 19]   #水体，蓝色，4

    cv2.imwrite(output_path,color)

color_annotation('data/train/1_class_8bits.png','train_2.png')

def train_statistic(img_class_path1='data/train/1_class_8bits.png',
                    img_class_path2='data/train/2_class_8bits.png'):
    '''
    统计类别个数
    '''
    img1 = misc.imread(img_class_path1)
    img2 = misc.imread(img_class_path2)
    other = np.sum(img1==0) + np.sum(img2==0)
    plant = np.sum(img1==1) + np.sum(img2==1)
    load =  np.sum(img1==2) + np.sum(img2==2)
    build = np.sum(img1==3) + np.sum(img2==3)
    water = np.sum(img1==4) + np.sum(img2==4)
    sum = other + plant + load + build + water

    print('other: %4f' % float(other/sum))
    print('plant: %4f' % float(plant/sum))
    print('load:  %4f' % float(load/sum))
    print('build: %4f' % float(build/sum))
    print('water: %4f' % float(water/sum))


