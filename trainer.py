import torch
import numpy as np
from torch.nn import DataParallel
from optimizer import get_optimizer
from modules import InPlaceABN, InPlaceABNSync
from net_io import load_and_save_net_model
import os, time


class Trainer:

    def __init__(self, warp, args):

        self.warp = warp

        self.args = args

        if isinstance(self.warp, DataParallel):

            self.net = self.warp.module.net

        else:

            self.net = self.warp.net

        self.lr_preset = np.array(args.lr_preset)

        self.lr_stage = np.array(args.lr_stage)

        self.optimizer = self.get_optimizer(args.optimizer_name)


    def train(self):
        pass

    def valiate(self):
        pass

    def get_optimizer(self, optimizer_name, used_look_head=True):

        return get_optimizer(optimizer_name , self.net, self.args, Trainer.learn_change_func, used_look_head=used_look_head)

    @staticmethod
    def learn_change_func(epoch, lr_preset, lr_stage):

        current_epoch_list = epoch > lr_stage
        current_epoch_ratio_index = np.sum(current_epoch_list)
        current_epoch_learn_ratio = lr_preset[current_epoch_ratio_index]

        return current_epoch_learn_ratio

    def do_train(self, args,  train_data_set, validate_data_set, save_dir ,start_epoch = 1):

        use_cuda = torch.cuda.is_available()
        self.net.train()

        for m in self.net.modules():
            if isinstance(m, InPlaceABNSync) or isinstance(m,InPlaceABN):
                if self.args.freeze:
                    m.eval()


        best_precision = 0.55

        for current_epoch in range(start_epoch,10000):

            loss_list = []
            precision = {}
            positive_accuracy_list = {}
            negative_accuracy_list = {}

            current_lr = Trainer.learn_change_func(current_epoch, np.array(self.args.lr_preset), np.array(self.args.lr_stage))

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            train_data_set.dataset.reset_cases()
            total_length = len(train_data_set)
            print(total_length)
            start_time = time.time()
            for i ,(data_and_mask, label, img_path) in enumerate(train_data_set):

                data_and_mask_cuda  =  data_and_mask.cuda().float()
                label_cuda          =  label.cuda().long()

                total_loss, accuracy_ratio_list, wrong_index, binary_predict_detach = self.warp(data_and_mask_cuda, label_cuda, train_stage = True)
                total_loss = torch.stack(total_loss).sum(dim=0)
                self.optimizer.zero_grad()

                loss_scalar = torch.mean(total_loss)

                loss_scalar.backward()

                self.optimizer.step()

                total_loss_detach = total_loss.mean().detach().cpu().numpy()
                loss_list.append(total_loss_detach)
                for idx, accuracy_ratio_ in enumerate(accuracy_ratio_list):
                    if idx not in precision:
                        precision[idx], positive_accuracy_list[idx], negative_accuracy_list[idx] = [], [], []
                        for idx2 in range(len(accuracy_ratio_)):
                            precision[idx].append([])
                            positive_accuracy_list[idx].append([])
                            negative_accuracy_list[idx].append([])
                    for idx2, accuracy_ratio in enumerate(accuracy_ratio_):
                        current_step_average_accuracy = np.mean(accuracy_ratio[0].float().detach().cpu().numpy())
                        current_step_average_positive_accuracy = np.nanmean(accuracy_ratio[1].detach().cpu().numpy())
                        current_step_average_negative_accuracy = np.nanmean(accuracy_ratio[2].detach().cpu().numpy())

                        precision[idx][idx2].append(current_step_average_accuracy)
                        positive_accuracy_list[idx][idx2].append(current_step_average_positive_accuracy)
                        negative_accuracy_list[idx][idx2].append(current_step_average_negative_accuracy)

                print('Out %d/%d loss %f  accuracy %s positive accuracy %s  negative accuracy %s' %(i,
                                                                                                    total_length,
                                                                                                    total_loss_detach,
                                                                                                    str({k: [item[-1] for item in v] for k,v in precision.items() if k == 0 or k == 1}),
                                                                                                    str({k: [item[-1] for item in v] for k,v in positive_accuracy_list.items() if k == 0 or k == 1}),
                                                                                                    str({k: [item[-1] for item in v] for k,v in negative_accuracy_list.items() if k == 0 or k == 1})))

            average_loss     = np.mean(np.array(loss_list))
            average_accuracy = {k: [np.mean(np.array(item)) for item in v] for k,v in precision.items()}
            average_positive_accuracy = {k: [np.nanmean(np.array(item)) for item in v] for k,v in positive_accuracy_list.items()}
            average_negative_accuracy = {k: [np.nanmean(np.array(item)) for item in v] for k,v in negative_accuracy_list.items()}



            print('Train Epoch %d lr %f average loss %f  average accuracy %s  average positive accuracy %s average negative accuracy %s, time: %s min'%(\
                current_epoch, current_lr, average_loss, str(average_accuracy), str(average_positive_accuracy), \
                str(average_negative_accuracy), str(int((time.time() - start_time)/60))))

            if current_epoch%args.save_freq == 0:
                load_and_save_net_model.netio_save(self.net, current_epoch,save_dir,args)

                print("Start Validate Process %d"%(current_epoch))
                start_time = time.time()

                validate_loss_list = []
                validate_precision = {}
                validate_positive_accuracy_list = {}
                validate_negative_accuracy_list = {}

                self.net.eval() #enter eval mode

                if not os.path.isdir(os.path.join(save_dir, 'pred')):
                    os.makedirs(os.path.join(save_dir, 'pred'))
                pred_txt = []
                for i, (data_and_mask, label, img_path) in enumerate(validate_data_set):

                    data_and_mask_cuda = data_and_mask.cuda().float()

                    label_cuda = label.cuda().long()

                    total_loss, accuracy_ratio_list, _, binary_predict_detach = self.warp(data_and_mask_cuda, label_cuda, train_stage=True)
                    binary_predict_detach = np.stack([item.cpu().numpy() for item in binary_predict_detach])
                    binary_predict_detach = binary_predict_detach.transpose((1, 0, 2))
                    for img_pathx, predx in zip(img_path, binary_predict_detach):
                        imgname = img_pathx.split('/')[-1].split('.')[0]
                        pred_txt.append(imgname + '\t' + ','.join([str(item) for item in predx[:, 1].tolist()]))
                        #np.save(os.path.join(save_dir, 'pred', imgname), predx)
                    total_loss = torch.stack(total_loss).sum(dim=0)
                    validate_loss_list.append(total_loss.mean().detach().cpu().numpy())

                    for idx, accuracy_ratio_ in enumerate(accuracy_ratio_list):
                        if idx not in validate_precision:
                            validate_precision[idx], validate_positive_accuracy_list[idx], validate_negative_accuracy_list[idx] = [], [], []
                            for idx2 in range(len(accuracy_ratio_)):
                                validate_precision[idx].append([])
                                validate_positive_accuracy_list[idx].append([])
                                validate_negative_accuracy_list[idx].append([])

                        for idx2, accuracy_ratio in enumerate(accuracy_ratio_):
                            current_step_validate_accuracy = accuracy_ratio[0].float().mean().detach().cpu().numpy()
                            current_step_validate_positive_accuracy = np.nanmean(accuracy_ratio[1].detach().cpu().numpy())
                            current_step_validate_negative_accuracy = np.nanmean(accuracy_ratio[2].detach().cpu().numpy())

                            validate_precision[idx][idx2].append(current_step_validate_accuracy)
                            validate_positive_accuracy_list[idx][idx2].append(current_step_validate_positive_accuracy)
                            validate_negative_accuracy_list[idx][idx2].append(current_step_validate_negative_accuracy)

                with open(os.path.join(save_dir, 'pred.txt'), 'w') as f:
                    f.write('\n'.join(pred_txt))
                average_loss = np.mean(np.array(validate_loss_list))
                average_accuracy = {k: [np.mean(np.array(item)) for item in v] for k,v in validate_precision.items()}
                average_positive_accuracy = {k: [np.nanmean(np.array(item)) for item in v] for k,v in validate_positive_accuracy_list.items()}
                average_negative_accuracy = {k: [np.nanmean(np.array(item)) for item in v] for k,v in validate_negative_accuracy_list.items()}

                self.net.train() #enter train mode

                print('Validate Epoch %d lr %f average loss %f  average accuracy %s average positive accuracy %s average negative accuracy %s, time: %s min' % (\
                    current_epoch, current_lr, average_loss, str(average_accuracy), str(average_positive_accuracy), \
                    str(average_negative_accuracy), str(int((time.time() - start_time)/60))))
    def crop_wrong_image(self, image, label, wrong_index, binary_predict_detach, idx, saved_dir):

        wrong_crop_image = image[wrong_index,:]

        wrong_label      = label[wrong_index]

        binary_predict    = np.max(binary_predict_detach, axis=1)
        binary_predict    = binary_predict[wrong_index]

        for index in range(len(wrong_label)):

            current_wrong_image = wrong_crop_image[index]
            current_gt_label   =  wrong_label[index]
            current_max_predict = binary_predict[index]

            idx += 1

            saved_name = "wrong_crop_image_" + str(idx) + "_" + str(current_gt_label) + "_" + str(current_max_predict) + ".npy"

            saved_name = os.path.join(saved_dir, saved_name)

            np.save(saved_name, current_wrong_image)


        return idx



    def do_test2(self, args, validate_data_set, save_dir, current_epoch):
        print("Start Validate Process %d"%(current_epoch))
        start_time = time.time()

        validate_loss_list = []
        validate_precision = {}
        validate_positive_accuracy_list = {}
        validate_negative_accuracy_list = {}

        self.net.eval() #enter eval mode

        if not os.path.isdir(os.path.join(save_dir, 'pred')):
            os.makedirs(os.path.join(save_dir, 'pred'))
        pred_txt = []
        for i, (data_and_mask, label, img_path) in enumerate(validate_data_set):

            data_and_mask_cuda = data_and_mask.cuda().float()

            label_cuda = label.cuda().long()

            total_loss, accuracy_ratio_list, _, binary_predict_detach = self.warp(data_and_mask_cuda, label_cuda, train_stage=True)
            binary_predict_detach = np.stack([item.cpu().numpy() for item in binary_predict_detach])
            binary_predict_detach = binary_predict_detach.transpose((1, 0, 2))
            for img_pathx, predx in zip(img_path, binary_predict_detach):
                imgname = img_pathx.split('/')[-1].split('.')[0]
                pred_txt.append(imgname + '\t' + ','.join([str(item) for item in predx[:, 1].tolist()]))
                np.save(os.path.join(save_dir, 'pred', imgname), predx)
            total_loss = torch.stack(total_loss).sum(dim=0)
            validate_loss_list.append(total_loss.mean().detach().cpu().numpy())

            for idx, accuracy_ratio_ in enumerate(accuracy_ratio_list):
                if idx not in validate_precision:
                    validate_precision[idx], validate_positive_accuracy_list[idx], validate_negative_accuracy_list[idx] = [], [], []
                    for idx2 in range(len(accuracy_ratio_)):
                        validate_precision[idx].append([])
                        validate_positive_accuracy_list[idx].append([])
                        validate_negative_accuracy_list[idx].append([])

                for idx2, accuracy_ratio in enumerate(accuracy_ratio_):
                    current_step_validate_accuracy = accuracy_ratio[0].float().mean().detach().cpu().numpy()
                    current_step_validate_positive_accuracy = np.nanmean(accuracy_ratio[1].detach().cpu().numpy())
                    current_step_validate_negative_accuracy = np.nanmean(accuracy_ratio[2].detach().cpu().numpy())

                    validate_precision[idx][idx2].append(current_step_validate_accuracy)
                    validate_positive_accuracy_list[idx][idx2].append(current_step_validate_positive_accuracy)
                    validate_negative_accuracy_list[idx][idx2].append(current_step_validate_negative_accuracy)


        with open(os.path.join(save_dir, 'pred.txt'), 'w') as f:
            f.write('\n'.join(pred_txt))
        average_loss = np.mean(np.array(validate_loss_list))
        average_accuracy = {k: np.mean(np.array(v)) for k,v in validate_precision.items()}
        average_positive_accuracy = {k: np.nanmean(np.array(v)) for k,v in validate_positive_accuracy_list.items()}
        average_negative_accuracy = {k: np.nanmean(np.array(v)) for k,v in validate_negative_accuracy_list.items()}

        print('Validate Epoch %d average loss %f  average accuracy %s average positive accuracy %s average negative accuracy %s, time: %s min' % (\
            current_epoch, average_loss, str(average_accuracy), str(average_positive_accuracy), \
            str(average_negative_accuracy), str(int((time.time() - start_time)/60))))




    def do_test(self, args, validate_data_set):

        use_cuda = torch.cuda.is_available()
        self.net.eval()

        for m in self.net.modules():
            if isinstance(m, InPlaceABNSync) or isinstance(m,InPlaceABN):
                if self.args.freeze:
                    m.eval()

        total_length = len(validate_data_set)

        print(total_length)

        print("Start Test Process")

        validate_loss_list = []

        validate_precision = []

        idx = 0

        for i, (data_and_mask, label) in enumerate(validate_data_set):

            data_and_mask_cuda = data_and_mask.cuda().float()

            label_cuda = label.cuda().long()

            total_loss, accuracy_ratio, wrong_index, binary_predict_detach = self.warp(data_and_mask_cuda, label_cuda, train_stage=True)

            # binary_out, multi_out = self.warp(data_and_mask_cuda, label_cuda, train_stage=False)

            # print(binary_out)
        #
            validate_precision.append(np.nanmean(accuracy_ratio[1].float().detach().cpu().numpy()))

            validate_loss_list.append(np.nanmean(total_loss.mean().detach().cpu().numpy()))

            print('out %d/%d Validate average loss %f  average accuracy %f'%(i, total_length, validate_loss_list[-1], validate_precision[-1]))

            if args.need_crop_wrong_img == True:

                data_and_mask_cuda = data_and_mask_cuda.cpu().numpy()
                label              = label.cpu().numpy()
                binary_predict_detach = binary_predict_detach.cpu().numpy()
                wrong_index = wrong_index.cpu().numpy()

                idx = self.crop_wrong_image(data_and_mask_cuda, label, wrong_index, binary_predict_detach, idx, args.crop_saved_dir)


        average_accuracy = np.nanmean(np.array(validate_precision))
        average_loss = np.nanmean(np.array(validate_loss_list))

        print('Validate average loss %f  average accuracy %f' % (average_loss, average_accuracy))





