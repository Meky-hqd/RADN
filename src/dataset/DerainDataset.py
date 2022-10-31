#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

from numpy.random import RandomState
from torch.utils.data import Dataset
from skimage import io, transform
import cv2
import h5py
import torch.utils.data as udata
from src.utils import *


# data_dir = "./rain100H"


class TrainDataset(Dataset):
    def __init__(self, data_dir, patch_size=128, is_trans=False):
        super(TrainDataset, self).__init__()
        self.rand_state = RandomState(66)  # 定义随机种子；
        self.data_dir = data_dir
        self.root_dir = self.data_dir
        self.root_dir_rain = os.path.join(self.root_dir, "rain")  # rain100L/train/rain
        self.root_dir_label = os.path.join(self.root_dir, "label")  # rain100L/train/val
        self.mat_files_rain = sorted(os.listdir(self.root_dir_rain))  # 将rain文件里的图像按命名序号大小排序
        self.mat_files_label = sorted(os.listdir(self.root_dir_label))
        self.patch_size = patch_size
        self.file_num = len(self.mat_files_label)  # train文件中雨图像的数目
        self.is_trans = is_trans

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name_rain = self.mat_files_rain[idx % self.file_num]

        file_name_label = self.mat_files_label[idx % self.file_num]  # 训练Rain100L的时候用这行；
        # file_name_label = file_name_rain.split('x')[0] + '.png'             # 训练Rain100H的时候用这行；       ‘x’  '.png'
        # file_name_label = file_name_rain.split('_')[0] + '.jpg'            #Rain1400的时候用这行      改为‘_’   '.jpg'    # 分割分隔符

        img_file_rain = os.path.join(self.root_dir_rain, file_name_rain)  # train/rain/norain-i.png
        img_file_label = os.path.join(self.root_dir_label, file_name_label)
        img_rain = io.imread(img_file_rain).astype(np.float32) / 255  # 读取图像，同时进行归一化；array数组的形式；
        img_label = io.imread(img_file_label).astype(np.float32) / 255
        if self.is_trans:
            O, B = self.trans(img_rain, img_label)
        else:
            O, B = self.crop(img_rain, img_label)  # O=img_rain    ， B=img_label
        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))  # numpy中的转置函数np.transpose()；
        return torch.Tensor(O), torch.Tensor(B)  # 将numpy表示的图像数据转化为Tensor张量形式；

    def trans(self, img_rain, img_label):
        patch_size = self.patch_size
        img_rain = transform.resize(img_rain, (patch_size, patch_size))
        img_label = transform.resize(img_label, (patch_size, patch_size))
        return img_rain, img_label

    def crop(self, img_rain, img_label):
        patch_size = self.patch_size
        h, w, c = img_rain.shape
        p_h, p_w = patch_size, patch_size
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_rain[r: r + p_h, c: c + p_w]
        B = img_label[r: r + p_h, c: c + p_w]
        return O, B


class ValDataset(Dataset):
    def __init__(self, data_dir):
        super(ValDataset, self).__init__()
        self.rand_state = RandomState(66)  # 定义随机种子；
        self.data_dir = data_dir
        self.root_dir = self.data_dir
        self.root_dir_rain = os.path.join(self.root_dir, "rain")  # rain100L/train/rain
        self.root_dir_label = os.path.join(self.root_dir, "label")  # rain100L/train/val
        self.mat_files_rain = sorted(os.listdir(self.root_dir_rain))  # 将rain文件里的图像按命名序号大小排序
        self.mat_files_label = sorted(os.listdir(self.root_dir_label))
        self.file_num = len(self.mat_files_label)  # train文件中雨图像的数目

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name_rain = self.mat_files_rain[idx % self.file_num]

        file_name_label = self.mat_files_label[idx % self.file_num]  # 训练Rain100L的时候用这行；
        # file_name_label = file_name_rain.split('x')[0] + '.png'             # 训练Rain100H的时候用这行；       ‘x’  '.png'
        # file_name_label = file_name_rain.split('_')[0] + '.jpg'            #Rain1400的时候用这行      改为‘_’   '.jpg'    # 分割分隔符

        img_file_rain = os.path.join(self.root_dir_rain, file_name_rain)  # train/rain/norain-i.png
        img_file_label = os.path.join(self.root_dir_label, file_name_label)
        img_rain = io.imread(img_file_rain).astype(np.float32) / 255  # 读取图像，同时进行归一化；array数组的形式；
        img_label = io.imread(img_file_label).astype(np.float32) / 255

        O, B = img_rain, img_label  # O=img_rain    ， B=img_label
        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))  # numpy中的转置函数np.transpose()；
        return torch.Tensor(O), torch.Tensor(B)  # 将numpy表示的图像数据转化为Tensor张量形式；

    def crop(self, img_rain, img_label):
        # 将输入的图像的长和宽修改为偶数
        H, W, C = img_rain.shape
        if H % 2 != 0:
            H = H - 1
        if W % 2 != 0:
            W = W - 1
        O = img_rain[:H, : W, :]
        B = img_rain[:H, : W, :]
        return O, B

    def trans(self, img_rain, img_label):
        patch_size = self.patch_size
        img_rain = transform.resize(img_rain, (patch_size, patch_size))
        img_label = transform.resize(img_label, (patch_size, patch_size))
        return img_rain, img_label
    # def crop(self, img_rain, img_label):
    #     patch_size = self.patch_size
    #     h, w, c = img_rain.shape
    #     p_h, p_w = patch_size, patch_size
    #     r = self.rand_state.randint(0, h - p_h)
    #     c = self.rand_state.randint(0, w - p_w)
    #     O = img_rain[r: r + p_h, c: c + p_w]
    #     B = img_label[r: r + p_h, c: c + p_w]
    #     return O, B


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data_Rain12600(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rainy_image')
    target_path = os.path.join(data_path, 'ground_truth')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        target = cv2.imread(os.path.join(target_path, target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(14):
            input_file = "%d_%d.jpg" % (i + 1, j + 1)
            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)
                train_num += 1

    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)


def prepare_data_RainTrainH(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rain')
    target_path = os.path.join(data_path, 'label')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path, target_file)):

            target = cv2.imread(os.path.join(target_path, target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "norain-%d.png" % (i + 1)

            if os.path.exists(os.path.join(input_path, input_file)):  # we delete 546 samples

                input_img = cv2.imread(os.path.join(input_path, input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_RainTrainL(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rain')
    target_path = os.path.join(data_path, 'label')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path, target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "norain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_Rain800(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rain')
    target_path = os.path.join(data_path, 'label')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(700):
        target_file = "%d.jpg" % (i + 1)
        target = cv2.imread(os.path.join(target_path, target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "%d.jpg" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_200(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path, target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "rain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 2:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


class Dataset(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset, self).__init__()

        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)


class TestDataset(udata.Dataset):
    def __init__(self, test_path):
        super().__init__()
        # self.rand_state = RandomState(66)
        self.input = os.path.join(os.path.join(test_path, 'rain'))
        self.target = os.path.join(os.path.join(test_path, 'label'))
        self.input_files = os.listdir(self.input)
        self.input_files.sort(key=lambda x: int(x[6:-4]), reverse=True)
        self.target_files = os.listdir(self.target)
        self.target_files.sort(key=lambda x: int(x[6:-4]), reverse=True)
        # self.patch_size = settings.patch_size
        self.file_num = len(self.target_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        input_name = self.input_files[idx % self.file_num]
        target_name = self.target_files[idx % self.file_num]
        input_file = os.path.join(self.input, input_name)
        target_file = os.path.join(self.target, target_name)
        input = cv2.imread(input_file).astype(np.float32) / 255
        b, g, r = cv2.split(input)
        input = cv2.merge([r, g, b])
        target = cv2.imread(target_file).astype(np.float32) / 255
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        return torch.Tensor(input.transpose(2, 0, 1)), torch.Tensor(target.transpose(2, 0, 1))

class TestDataset(udata.Dataset):
    def __init__(self, test_path):
        super().__init__()
        # self.rand_state = RandomState(66)
        self.input = os.path.join(os.path.join(test_path, 'rain'))
        self.target = os.path.join(os.path.join(test_path, 'label'))
        self.input_files = os.listdir(self.input)
        self.input_files.sort(key=lambda x: int(x[6:-4]), reverse=True)
        self.target_files = os.listdir(self.target)
        self.target_files.sort(key=lambda x: int(x[6:-4]), reverse=True)
        # self.patch_size = settings.patch_size
        self.file_num = len(self.target_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        input_name = self.input_files[idx % self.file_num]
        target_name = self.target_files[idx % self.file_num]
        input_file = os.path.join(self.input, input_name)
        target_file = os.path.join(self.target, target_name)
        input = cv2.imread(input_file).astype(np.float32) / 255
        b, g, r = cv2.split(input)
        input = cv2.merge([r, g, b])
        target = cv2.imread(target_file).astype(np.float32) / 255
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        return torch.Tensor(input.transpose(2, 0, 1)), torch.Tensor(target.transpose(2, 0, 1))


class TestDataset800(udata.Dataset):
    def __init__(self, test_path):
        super().__init__()
        # self.rand_state = RandomState(66)
        self.input = os.path.join(os.path.join(test_path, 'rain'))
        self.target = os.path.join(os.path.join(test_path, 'label'))
        self.input_files = os.listdir(self.input)
        self.input_files.sort(key=lambda x: int(x.split('.')[0]), reverse=True)
        self.target_files = os.listdir(self.target)
        self.target_files.sort(key=lambda x: int(x.split('.')[0]), reverse=True)
        # self.patch_size = settings.patch_size
        self.file_num = len(self.target_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        input_name = self.input_files[idx % self.file_num]
        target_name = self.target_files[idx % self.file_num]
        input_file = os.path.join(self.input, input_name)
        target_file = os.path.join(self.target, target_name)
        input = cv2.imread(input_file).astype(np.float32) / 255
        b, g, r = cv2.split(input)
        input = cv2.merge([r, g, b])
        target = cv2.imread(target_file).astype(np.float32) / 255
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        return torch.Tensor(input.transpose(2, 0, 1)), torch.Tensor(target.transpose(2, 0, 1))


class metric_TestDataset(udata.Dataset):
    def __init__(self, norain_path, derain_path):
        super().__init__()
        # self.rand_state = RandomState(66)
        self.input = os.path.join(derain_path)
        self.target = os.path.join(norain_path)
        self.input_files = os.listdir(self.input)
        # self.input_files.sort(key=lambda x:int(x[:-4]))
        self.target_files = os.listdir(self.target)
        # self.target_files.sort(key=lambda x:int(x[:-4]))
        # self.patch_size = settings.patch_size
        self.file_num = len(self.target_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        input_name = self.input_files[idx % self.file_num]
        target_name = self.target_files[idx % self.file_num]
        input_file = os.path.join(self.input, input_name)
        target_file = os.path.join(self.target, target_name)
        input = cv2.imread(input_file).astype(np.float32) / 255
        b, g, r = cv2.split(input)
        input = cv2.merge([r, g, b])
        target = cv2.imread(target_file).astype(np.float32) / 255
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        return torch.Tensor(input.transpose(2, 0, 1)), torch.Tensor(target.transpose(2, 0, 1))