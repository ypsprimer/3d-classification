import os
import time
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate, binary_dilation
import random
import math


class ClassBasicDataSet(Dataset):

    def __init__(self, args, image_root_dir, used_list_txt, phase):

        assert phase in ['train', 'val', 'test']

        self.phase = phase

        self.args = args

        self.crop_size = self.args.crop_size

        self.negative_ratio = self.args.negative_ratio

        with open(used_list_txt, 'r') as f:
            self.cases = [line.strip().split() for line in f.readlines()]

        self.image_cases_all = [[os.path.join(image_root_dir, item[0] + '.npy'), int(item[1]), float(item[2]), int(item[4]), float(item[5])] for item in self.cases]

        self.image_cases_pos = [item for item in self.image_cases_all if item[1] == 1 and item[3] < 5 and item[4] == 1]


        self.image_cases_pos_incls = {i: [] for i in range(5)}
        for item in self.image_cases_pos:
            self.image_cases_pos_incls[item[3]].append(item)
        print('pos samples: ')
        print(str({k: len(v) for k,v in self.image_cases_pos_incls.items()}))

        self.image_cases_neg = [item for item in self.image_cases_all if item[1] == 0]

        print('all: ', len(self.image_cases_all), ', pos: ', len(self.image_cases_pos), ', neg: ', len(self.image_cases_neg))

        self.image_cases = self.image_cases_pos
        # print(self.image_cases)
        self.base_size = 64

        self.sample_ratio_cls = [0.3, 2, 1, 1, 1]

    def reset_cases(self):
        print('resample pos samples...')
        self.image_cases_pos_samples = []
        for cls, samples in self.image_cases_pos_incls.items():
            random.shuffle(samples)
            sample_ratio = self.sample_ratio_cls[cls]
            sample_count = int(len(samples) * sample_ratio)
            samples_tmp = samples * math.ceil(sample_ratio)
            self.image_cases_pos_samples += samples_tmp[: sample_count]
            print(cls, len(samples_tmp[: sample_count]))
        print(len(self.image_cases_pos_samples))
        print('pos: ', self.image_cases_pos_samples[:5])
        print('shuffle neg samples...')
        print('before shuffle: ', self.image_cases_neg[:5])
        random.shuffle(self.image_cases_neg)
        print('after shuffle: ', self.image_cases_neg[:5])
        self.image_cases = self.image_cases_pos_samples #+ self.image_cases_neg[: len(self.image_cases_pos) * self.negative_ratio]

    def __getitem__(self, idx):

        t = time.time()

        np.random.seed(int(str(t%1)[2:7]))#seed according to time

        current_image_path, current_label, current_bbox_size, current_cls, conf = self.image_cases[idx]

        current_image = np.load(current_image_path)
        nodule_part_ratio = current_bbox_size / current_image.shape[0]
        assert current_image.shape[0] >= self.base_size
        if np.any(np.array(current_image.shape) > self.base_size):
            current_image = zoom(current_image, self.base_size / np.array(current_image.shape).astype('float'), order=0)

        current_mask = np.zeros_like(current_image)
        bbox_start_zxy = current_image.shape[0] * (1 - nodule_part_ratio) / 2
        bbox_size = current_image.shape[0] * nodule_part_ratio
        bbox_start_zxy, bbox_size = np.around(bbox_start_zxy).astype('int32'), np.around(bbox_size).astype('int32')
        current_mask[bbox_start_zxy: bbox_start_zxy + bbox_size, \
                     bbox_start_zxy: bbox_start_zxy + bbox_size, \
                     bbox_start_zxy: bbox_start_zxy + bbox_size] = 1
        current_image, current_mask = current_image[np.newaxis], current_mask[np.newaxis]

        flip   = self.args.flip
        ifswap = self.args.ifswap

        if self.phase == "train":

            image_augmentation, mask_augmentation =  self.data_augmentation(current_image, current_mask, flip, ifswap)
            # image_augmentation, mask_augmentation = current_image, current_mask

        else:

            image_augmentation, mask_augmentation = current_image, current_mask

        image_augmentation = self.image_normalization(image_augmentation)

        image_mask = np.concatenate((image_augmentation, mask_augmentation), 0)

        #print(image_mask.shape, current_label, current_image_path.split('/')[-1])

        label = np.zeros((5,), dtype='int64')
        if current_label == 1:
            label[current_cls] = 1

        return image_mask, label, current_image_path



    def __len__(self):

        return len(self.image_cases)

    def image_normalization(self, image, type="USM"):

        image = image/100

        return image

    def data_augmentation(self, image, mask, flip, ifswap):
        '''
        :param image:  image [1 z y x ]
        :param mask:   mask [1 , z ,y x]
        :param flip:   wether flip
        :param ifswap: wether ifswap
        :return:
        '''

        if flip == True and self.phase == "train":

            flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1

            image = np.ascontiguousarray(image[:, :, ::flipid[1], ::flipid[2]])
            mask = np.ascontiguousarray(mask[:, :, ::flipid[1], ::flipid[2]])

        if ifswap == True and self.phase == "train":

            if image.shape[1]==image.shape[2] and image.shape[1]==image.shape[3]:
                axisorder = np.random.permutation(3)
                image = np.transpose(image, np.concatenate([[0], axisorder + 1]))
                mask = np.transpose(mask, np.concatenate([[0], axisorder + 1]))

        # print(mask[0,24,12:36,12:36])


        return image , mask


class ClassifierDataSet(ClassBasicDataSet):

    def __init__(self, args, image_root_dir, used_list_txt, phase):

        super(ClassifierDataSet, self).__init__(args, image_root_dir, used_list_txt, phase)

        # self.label_mapping = LabelMapping(config, self.phase, 'stage1')
        # if self.phase != 'test':
        #     self.bboxes = []
        #     for i, l in enumerate(self.sample_bboxes):
        #         if len(l) > 0 :
        #             for t in l:
        #                 self.bboxes.append([1,i,t])



