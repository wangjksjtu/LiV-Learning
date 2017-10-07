import scipy.misc
import random
import numpy as np
from math import ceil

class DVR_Provider:
    def __init__(self, input_dir='data/DVR/'):
        self.xs = []
        self.ys1 = [] # vehicle speeds
        self.ys2 = [] # wheel angles
        self.ys = []
        self.train_pointer = 0
        self.val_pointer = 0
        self.path = input_dir
        self.read()
        self.num_images = len(self.xs)
        self.transform()
        self.split_set(0.8)
        self.suffle()

    def read(self, filename="data.txt"):
        with open(self.path + filename) as f:
            for line in f:
                self.xs.append(self.path + line.split()[0])
                self.ys1.append((float(line.split()[1]) -20) / 20)
                self.ys2.append(float(line.split()[2]) * scipy.pi / 180)

    def transform(self):
        c = list(zip(self.xs, self.ys1, self.ys2))
        xs, ys1, ys2 = zip(*c)
        self.xs = np.asarray(xs)
        self.ys = np.transpose(np.asarray((ys1, ys2)))

    def suffle(self):
        c = list(zip(self.train_xs, self.train_ys))
        random.shuffle(c)
        self.train_xs, self.train_ys = zip(*c)

        c = list(zip(self.val_xs, self.val_ys))
        random.shuffle(c)
        self.val_xs, self.val_ys = zip(*c)

    def split_set(self, rate = 0.8):
        self.train_xs = self.xs[: int(rate * self.num_images)]
        self.train_ys = self.ys[: int(rate * self.num_images)]
        self.val_xs = self.xs[-int((1 - rate) * self.num_images) :]
        self.val_ys = self.ys[-int((1 - rate) * self.num_images) :]

    def load_one_batch(self, batch_size, Type='train', shape=[66, 200]):
        x_out = []
        y_out = []
        if Type == 'train':
            for i in range(0, batch_size):
                index = (self.train_pointer + i) % len(self.train_xs)
                x_out.append(scipy.misc.imresize(scipy.misc.imread(
                self.train_xs[index]), shape) / 255.0)
                y_out.append(self.train_ys[index])
                self.train_pointer += batch_size
        else:
            for i in range(0, batch_size):
                index = (self.val_pointer + i) % len(self.val_xs)
                x_out.append(scipy.misc.imresize(scipy.misc.imread(
                self.val_xs[index]), shape) / 255.0)
                y_out.append(self.val_ys[index])
                self.val_pointer += batch_size
        return np.stack(x_out), np.stack(y_out)

    def load_val_all(self, batch_size, shape = [66, 200]):
        x_out = []
        y_out = []
        index = 0
        iteration = int(ceil(len(self.val_xs) / float(batch_size)))
        #print iteration
        for i in range(iteration):
            xs = []
            ys = []
            if i == iteration - 1:
                for i in range(index, len(self.val_xs)):
                    xs.append(scipy.misc.imresize(scipy.misc.imread(
                    self.val_xs[i]), shape) / 255.0)
                    ys.append(self.val_ys[i])
            else:
                for i in range(0, batch_size):
                    xs.append(scipy.misc.imresize(scipy.misc.imread(
                    self.val_xs[index + i]), shape) / 255.0)
                    ys.append(self.val_ys[index + i])
                index += batch_size

            x_out.append(np.stack(xs))
            y_out.append(np.stack(ys))
        return np.asarray(x_out), np.asarray(y_out)

class DVR_FMAP_Provider:
    def __init__(self, input_dir1='data/DVR/', input_dir2='data/FMAP/'):
        self.xs1 = []
        self.xs2 = []
        self.ys1 = [] # vehicle speeds
        self.ys2 = [] # wheel angles
        self.ys = []
        self.train_pointer = 0
        self.val_pointer = 0
        self.input_dir1 = input_dir1
        self.input_dir2 = input_dir2
        self.read()
        self.num_images = len(self.xs1)
        self.transform()
        self.split_set(0.8)
        self.suffle()

    def read(self, filename="data.txt"):
        with open(self.input_dir1 + filename) as f:
            for line in f:
                self.xs1.append(self.input_dir1 + line.split()[0])
                self.xs2.append(self.input_dir2 + line.split()[0])
                self.ys1.append((float(line.split()[1]) -20) / 20)
                self.ys2.append(float(line.split()[2]) * scipy.pi / 180)

    def transform(self):
        c = list(zip(self.xs1, self.xs2, self.ys1, self.ys2))
        xs1, xs2, ys1, ys2 = zip(*c)
        self.xs1, self.xs2 = np.asarray(xs1), np.asarray(xs2)
        self.ys = np.transpose(np.asarray((ys1, ys2)))

    def suffle(self):
        c = list(zip(self.train_xs1, self.train_xs2, self.train_ys))
        random.shuffle(c)
        self.train_xs1, self.train_xs2, self.train_ys = zip(*c)

        c = list(zip(self.val_xs1, self.val_xs2, self.val_ys))
        random.shuffle(c)
        self.val_xs1, self.val_xs2, self.val_ys = zip(*c)

    def split_set(self, rate = 0.8):
        self.train_xs1 = self.xs1[: int(rate * self.num_images)]
        self.train_xs2 = self.xs2[: int(rate * self.num_images)]
        self.train_ys = self.ys[: int(rate * self.num_images)]
        self.val_xs1 = self.xs1[-int((1 - rate) * self.num_images):]
        self.val_xs2 = self.xs2[-int((1 - rate) * self.num_images):]
        self.val_ys = self.ys[-int((1 - rate) * self.num_images):]

    def load_one_batch(self, batch_size, Type='train', shape1=[66, 200], shape2=[66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        if Type == 'train':
            for i in range(0, batch_size):
                index = (self.train_pointer + i) % len(self.train_xs1)
                x_out1.append(scipy.misc.imresize(scipy.misc.imread(
                self.train_xs1[index]), shape1) / 255.0)
                x_out2.append(scipy.misc.imresize(scipy.misc.imread(
                self.train_xs2[index]), shape2) / 255.0)
                y_out.append(self.train_ys[index])
                self.train_pointer += batch_size
        else:
            for i in range(0, batch_size):
                index = (self.val_pointer + i) % len(self.val_xs1)
                x_out1.append(scipy.misc.imresize(scipy.misc.imread(
                self.val_xs1[index]), shape1) / 255.0)
                x_out2.append(scipy.misc.imresize(scipy.misc.imread(
                self.val_xs2[index]), shape2) / 255.0)
                y_out.append(self.val_ys[index])
                self.val_pointer += batch_size
        return np.stack(x_out1), np.stack(x_out2), np.stack(y_out)

    def load_val_all(self, batch_size, shape1 = [66, 200], shape2 = [66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        index = 0
        iteration = int(ceil(len(self.val_xs1) / float(batch_size)))
        for i in range(iteration):
            xs1 = []
            xs2 = []
            ys = []
            if i == iteration - 1:
                for i in range(index, len(self.val_xs1)):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                    self.val_xs1[i]), shape1) / 255.0)
                    xs2.append(scipy.misc.imresize(scipy.misc.imread(
                    self.val_xs2[i]), shape2) / 255.0)
                    ys.append(self.val_ys[i])
            else:
                for i in range(0, batch_size):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                    self.val_xs1[index + i]), shape1) / 255.0)
                    xs2.append(scipy.misc.imresize(scipy.misc.imread(
                    self.val_xs2[index + i]), shape2) / 255.0)
                    ys.append(self.val_ys[index + i])
                index += batch_size

            x_out1.append(np.stack(xs1))
            x_out2.append(np.stack(xs2))
            y_out.append(np.stack(ys))
        return np.asarray(x_out1), np.asarray(x_out2), np.asarray(y_out)

class DVR_Points_Provider:
    def __init__(self, input_dir1='data/DVR/', input_dir2='data/points_16192/'):
        self.xs1 = [] # DVR
        self.xs2 = [] # points
        self.ys1 = [] # vehicle speeds
        self.ys2 = [] # wheel angles
        self.ys = []
        self.train_pointer = 0
        self.val_pointer = 0
        self.input_dir1 = input_dir1
        self.input_dir2 = input_dir2
        self.read()
        self.num_images = len(self.xs1)
        self.transform()
        self.split_set(0.8)
        self.suffle()

    def read(self, filename="data.txt"):
        with open(self.input_dir1 + filename) as f:
            for line in f:
                self.xs1.append(self.input_dir1 + line.split()[0])
                self.xs2.append(self.input_dir2 + line.split()[0])
                self.ys1.append((float(line.split()[1]) -20) / 20)
                self.ys2.append(float(line.split()[2]) * scipy.pi / 180)

    def transform(self):
        c = list(zip(self.xs1, self.xs2, self.ys1, self.ys2))
        xs1, xs2, ys1, ys2 = zip(*c)
        self.xs1, self.xs2 = np.asarray(xs1), np.asarray(xs2)
        self.ys = np.transpose(np.asarray((ys1, ys2)))

    def suffle(self):
        c = list(zip(self.train_xs1, self.train_xs2, self.train_ys))
        random.shuffle(c)
        self.train_xs1, self.train_xs2, self.train_ys = zip(*c)

        c = list(zip(self.val_xs1, self.val_xs2, self.val_ys))
        random.shuffle(c)
        self.val_xs1, self.val_xs2, self.val_ys = zip(*c)

    def split_set(self, rate = 0.8):
        self.train_xs1 = self.xs1[: int(rate * self.num_images)]
        self.train_xs2 = self.xs2[: int(rate * self.num_images)]
        self.train_ys = self.ys[: int(rate * self.num_images)]
        self.val_xs1 = self.xs1[-int((1 - rate) * self.num_images):]
        self.val_xs2 = self.xs2[-int((1 - rate) * self.num_images):]
        self.val_ys = self.ys[-int((1 - rate) * self.num_images):]

    def load_one_batch(self, batch_size, Type='train', shape=[66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        if Type == 'train':
            for i in range(0, batch_size):
                index = (self.train_pointer + i) % len(self.train_xs1)
                x_out1.append(scipy.misc.imresize(scipy.misc.imread(
                self.train_xs1[index]), shape) / 255.0)
                x_out2.append(np.loadtxt(self.train_xs2[index], delimiter=','))
                y_out.append(self.train_ys[index])
                self.train_pointer += batch_size
        else:
            for i in range(0, batch_size):
                index = (self.val_pointer + i) % len(self.val_xs)
                x_out1.append(scipy.misc.imresize(scipy.misc.imread(
                self.val_xs1[index]), shape) / 255.0)
                x_out2.append(np.loadtxt(self.val_xs2[index], delimiter=','))
                y_out.append(self.val_ys[index])
                self.val_pointer += batch_size
        return np.stack(x_out1), np.stack(x_out2), np.stack(y_out)

    def load_val_all(self, batch_size, shape = [66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        index = 0
        iteration = int(ceil(len(self.val_xs1) / float(batch_size)))
        for i in range(iteration):
            xs1 = []
            xs2 = []
            ys = []
            if i == iteration - 1:
                for i in range(index, len(self.val_xs1)):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                    self.val_xs1[i]), shape) / 255.0)
                    xs2.append(np.loadtxt(self.val_xs2[i], delimiter=','))
                    ys.append(self.val_ys[i])
            else:
                for i in range(0, batch_size):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                    self.val_xs1[index + i]), shape1) / 255.0)
                    xs2.append(np.loadtxt(self.val_xs2[index + i], delimiter=','))
                    ys.append(self.val_ys[index + i])
                index += batch_size

            x_out1.append(np.stack(xs1))
            x_out2.append(np.stack(xs2))
            y_out.append(np.stack(ys))
        return np.asarray(x_out1), np.asarray(x_out2), np.asarray(y_out)

if __name__ == "__main__":
    '''
    instance1 = DVR_Provider()
    print len(instance1.xs), len(instance1.ys)
    print len(instance1.train_xs), len(instance1.train_ys)
    print len(instance1.val_xs), len(instance1.val_ys)
    print instance1.val_xs[:3], instance1.val_ys[:3]
    x_, y_ = instance1.load_one_batch(16, 'train')
    print x_.shape, y_.shape
    x_, y_ = instance1.load_one_batch(16, 'val')
    print x_.shape, y_.shape
    x_, y_ = instance1.load_val_all(16)
    print len(x_), len(y_)
    print x_[0].shape
    print x_[len(x_) - 1].shape

    instance2 = DVR_FMAP_Provider()
    print len(instance2.xs1), len(instance2.xs2), len(instance2.ys)
    print len(instance2.train_xs1), len(instance2.train_xs2), len(instance2.train_ys)
    print len(instance2.val_xs1), len(instance2.val_xs2), len(instance2.val_ys)
    print instance2.val_xs1[:3], instance2.val_xs2[:3], instance2.val_ys[:3]
    for i in range(1):
        x1_, x2_, y_ = instance2.load_one_batch(156, 'train')
        print x1_.shape, x2_.shape, y_.shape
    x1_, x2_, y_ = instance2.load_val_all(16)
    print len(x1_), len(x2_), len(y_)
    print x1_[0].shape
    print x2_[len(x_) - 1].shape
    print y_[2].shape
    '''
    instance3 = DVR_Points_Provider()
    print len(instance2.xs1), len(instance2.xs2), len(instance2.ys)
    print len(instance2.train_xs1), len(instance2.train_xs2), len(instance2.train_ys)
    print len(instance2.val_xs1), len(instance2.val_xs2), len(instance2.val_ys)
    print instance2.val_xs1[:3], instance2.val_xs2[:3], instance2.val_ys[:3]
