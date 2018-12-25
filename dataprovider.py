from scipy import misc
from tqdm import tqdm
import numpy as np



'将生成batch等多个方法封装成一个类'
class Dataprovider():
    '初始化'
    def __init__(self,
                 train_list='train.txt',
                 test_list='test.txt',):
        self.train_list = train_list
        self.test_list = test_list
        self.x_tr, self.y_tr = self.read_train()
        self.x_test, self.y_test = self.read_test()
        self.batch_offset_tr = 0
        self.batch_offset_test = 0
        self.y_tr = np.expand_dims(self.y_tr,-1)
        self.y_test = np.expand_dims(self.y_test,-1)

    def read_train(self):
        x_tr = []
        y_tr = []
        list = []
        with open(self.train_list) as f:
            for lines in f:
                list.append(lines)
        for i in tqdm(list):
            x_tr.append(self.read(i.split()[0]))
            y_tr.append(self.read(i.split()[1]))

        return np.array(x_tr), np.array(y_tr)

    def read_test(self):
        x_test = []
        y_test = []
        list = []
        with open(self.test_list) as f:
            for lines in f:
                list.append(lines)
        for i in tqdm(list):
            x_test.append(self.read(i.split()[0]))
            y_test.append(self.read(i.split()[1]))
        return np.array(x_test), np.array(y_test)

    def read(self, path):
        return misc.imread(path)

    def get_train(self):
        return self.x_tr, self.y_tr

    def get_test(self):
        return self.x_test, self.y_test



    def next_batch_tr(self, batch_size):
        start = self.batch_offset_tr
        self.batch_offset_tr += batch_size
        if self.batch_offset_tr > self.x_tr.shape[0]:
            # Shuffle the data
            perm = np.arange(self.x_tr.shape[0])
            np.random.shuffle(perm)
            self.x_tr = self.x_tr[perm]
            self.y_tr = self.y_tr[perm]
            start = 0
            self.batch_offset_tr = batch_size
        end = self.batch_offset_tr
        x_tr = self.x_tr[start:end]
        y_tr = self.y_tr[start:end]
        return x_tr, y_tr

    def next_batch_test(self, batch_size):
        start = self.batch_offset_test
        self.batch_offset_test += batch_size
        if self.batch_offset_test > self.x_test.shape[0]:
            # Shuffle the data
            perm = np.arange(self.x_test.shape[0])
            np.random.shuffle(perm)
            self.x_test = self.x_test[perm]
            self.y_test = self.y_test[perm]
            start = 0
            self.batch_offset_test = batch_size
        end = self.batch_offset_test
        x_test = self.x_test[start:end]
        y_test = self.y_test[start:end]
        return x_test, y_test
