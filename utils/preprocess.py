import os
from tqdm import tqdm
import numpy as np
import cv2

img_w = 256
img_h = 256



def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    for i in range(200):
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb


image_sets = ['data/train/1.png', 'data/train/2.png']
label_sets = ['data/train/1_class.png', 'data/train/2_class.png']

def creat_dataset():
    print('creating dataset...')
    g_count = 0
    for i in range(2):
        src_img = cv2.imread(image_sets[i])
        label_img = cv2.imread(label_sets[i], cv2.IMREAD_GRAYSCALE)
        num_h = (src_img.shape[0] - 256) // 128
        num_w = (src_img.shape[1] - 256) // 128
        for h in tqdm(range(num_h)):
            for w in range(num_w):
                for times in range(8):
                    sub_image = src_img[(h * 128):(h * 128 + 256), (w * 128):(w * 128 + 256), :]
                    sub_label = label_img[(h * 128):(h * 128 + 256), (w * 128):(w * 128 + 256)]
                    sub_image,sub_label = data_augment(sub_image, sub_label)
                    cv2.imwrite(('dataset/train/images/%05d.png' % g_count), sub_image)
                    cv2.imwrite(('dataset/train/labels/%05d.png' % g_count), sub_label)
                    g_count += 1

def creat_testdataset():
    g_count = 0
    src_img = cv2.imread('data/test/test3.png')
    label_img = cv2.imread('data/test/test3_labels_8bits.png',cv2.IMREAD_GRAYSCALE)
    num_h = (src_img.shape[0] - 256) // 128
    num_w = (src_img.shape[1] - 256) // 128
    for h in tqdm(range(num_h)):
        for w in range(num_w):
            sub_image = src_img[(h * 128):(h * 128 + 256), (w * 128):(w * 128 + 256), :]
            sub_label = label_img[(h * 128):(h * 128 + 256), (w * 128):(w * 128 + 256)]
            cv2.imwrite(('dataset/test/images/%05d.png' % g_count), sub_image)
            cv2.imwrite(('dataset/test/labels/%05d.png' % g_count), sub_label)
            g_count += 1


def generatepathlist(train_path='dataset/train/',
                     test_path='dataset/test/'):
    with open('train.txt','w') as f:
        list = os.listdir(train_path+'images/')
        for i in range(len(list)):
            image_path = train_path + 'images/' + list[i]
            label_path = train_path + 'labels/' + list[i]
            f.write(image_path+' '+label_path+'\n')
    with open('test.txt','w') as f:
        list = os.listdir(test_path+'images/')
        for i in range(len(list)):
            image_path = test_path + 'images/' + list[i]
            label_path = test_path + 'labels/' + list[i]
            f.write(image_path+' '+label_path+'\n')

if __name__ == '__main__':
    creat_dataset()
    creat_testdataset()
    generatepathlist()