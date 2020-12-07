import os
import time
import numpy as np
from torch.utils.data import Dataset


class ClassBasicDataSet(Dataset):

    def __init__(self, args, image_root_dir, used_list_txt, phase):

        assert phase in ['train', 'val', 'test']

        self.phase = phase

        self.args = args

        self.crop_size = self.args.crop_size

        self.negative_ratio = self.args.negative_ratio

        with open(used_list_txt, 'r') as f:
            self.cases = f.readlines()

        self.image_cases = [[os.path.join(image_root_dir,f.split('\n')[0].split(" ")[0]), int(f.split('\n')[0].split(" ")[1])] for f in self.cases]

        # print(self.image_cases)

    def __getitem__(self, idx):

        t = time.time()

        np.random.seed(int(str(t%1)[2:7]))#seed according to time

        current_image_path , current_label = self.image_cases[idx]

        current_image_mask = np.load(current_image_path)
        current_image = current_image_mask[0:self.crop_size,:,:][np.newaxis].astype(np.float32)
        current_mask  = current_image_mask[self.crop_size::,:,:][np.newaxis].astype(np.float32)

        flip   = self.args.flip
        ifswap = self.args.ifswap

        if self.phase == "train":

            image_augmentation, mask_augmentation =  self.data_augmentation(current_image, current_mask, flip, ifswap)
            # image_augmentation, mask_augmentation = current_image, current_mask

        else:

            image_augmentation, mask_augmentation = current_image, current_mask

        image_augmentation = self.image_normalization(image_augmentation)

        image_mask = np.concatenate((image_augmentation, mask_augmentation), 0)

        return image_mask, current_label,



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



