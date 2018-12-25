from scipy import misc
from tqdm import tqdm
import numpy as np
import cv2
wrong_label = misc.imread('data/test/test3_labels_8bits.png')
label = np.zeros(wrong_label.shape)

label[wrong_label==1] = 1
label[wrong_label==3] = 4
label[wrong_label==2] = 3
label[wrong_label==4] = 2


cv2.imwrite('data/test3_labels_8bits.png',label)