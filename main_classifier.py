import argparse
import os
import numpy as np
import torch
from net_io import load_and_save_net_model
from config import config_classifier
# from classifier_net import get_model
from classifier_densenet2 import get_model
# from classifier_senet import get_model
from torch.utils.data import DataLoader
from torch.nn import DataParallel


import datetime
import sys
import shutil
from torch.backends import cudnn
from wraper import Wraper
from trainer import Trainer
from net_io import message_logger
from dataer2 import ClassifierDataSet


parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Classifier')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch_size', default=64, type=int,metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--gpu', default=[0,1,2,3,4,5,6,7], type=list, metavar='N',help='use gpu')

# parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 32)')
# parser.add_argument('-b', '--batch_size', default=1, type=int,metavar='N', help='mini-batch size (default: 16)')
# parser.add_argument('--gpu', default=[0], type=list, metavar='N',help='use gpu')

parser.add_argument('--model', '-m', metavar='MODEL', default='base',help='model')
parser.add_argument('--optimizer_name', default='adam',help='optimizer name')
parser.add_argument('--copy_file', default=True, type=bool, help='copy file to saved dir')
parser.add_argument('--negative_ratio', default=4, type=int, help='negative : positive')
parser.add_argument('--flip', default=True, type=bool, help='data augmentation flip')
parser.add_argument('--ifswap', default=True, type=bool, help='data augmentation flip')
parser.add_argument('--crop_size', default=48, type=int, help='crop size')
parser.add_argument('--multiple_channel', default=6, type=int, help='multiple label')
parser.add_argument('--calculate_multi_label_loss', default=False, type=bool, help='wether calculate multiple loss')

parser.add_argument('--need_crop_wrong_img', default=True, type=bool, help='wether crop image when test mode')
parser.add_argument('--crop_saved_dir', default="/data_ssd/lxw/LungNodule/mvd_0902_wrong_crop_image", type=str, help='saved crop image dir')


parser.add_argument('--freeze', action = 'store_true', help='freeze bn')


parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save_freq', default=5, type=int, metavar='S',help='save frequency')
parser.add_argument('--save_dir', default='', type=str, metavar='SAVE',help='directory to save checkpoint (default: none)')

parser.add_argument('--resume', default='/workspace/lxw/alg_lung_nodule_classfier/results/base-2019925/0.852.ckpt', type=str, metavar='PATH',help='path to latest checkpoint (default: none)') #sgd
# parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)') #sgd

parser.add_argument('--train_mode', default="test", type=str, help='train or test')

parser.add_argument('--preprocess_path_train', default="/data_ssd/lxw/LungNodule/mvd_0902_classifier_small_crop_image_train/",type=str, help="train data preprocessed path")
parser.add_argument('--preprocess_path_valid', default="/data_ssd/lxw/LungNodule/mvd_0902_classifier_small_crop_image_validate/",type=str, help="validate data preprocessed path")
parser.add_argument('--train_splite_txt_path',    default="/data_ssd/lxw/LungNodule/mvd_0902_classifier_small_crop_image_train/train_splite.txt",type=str, help="train splite txt")
parser.add_argument('--validate_splite_txt_path', default="/data_ssd/lxw/LungNodule/mvd_0902_classifier_small_crop_image_validate/validate_splite.txt",type=str, help="valid splite txt")

parser.add_argument('--preprocess_path_test', default="/data_ssd/lxw/LungNodule/mvd_0902_classifier_small_crop_image_train/",type=str, help="train data preprocessed path")
parser.add_argument('--test_splite_txt_path',    default="/data_ssd/lxw/LungNodule/mvd_0902_classifier_small_crop_image_train/train_splite.txt",type=str, help="train splite txt")

# parser.add_argument('--preprocess_path_train', default="/workspace/tmp/",type=str, help="train data preprocessed path")
# parser.add_argument('--preprocess_path_valid', default="/workspace/tmp/",type=str, help="validate data preprocessed path")
# parser.add_argument('--train_splite_txt_path',    default="/workspace/tmp/test_train.txt",type=str, help="train splite txt")
# parser.add_argument('--validate_splite_txt_path', default="/workspace/tmp/test_train.txt",type=str, help="valid splite txt")


def main():

    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)

    torch.cuda.set_device(0)

    config = config_classifier.config

    args.lr_stage = config['lr_stage']

    args.lr_preset = config['lr']

    start_epoch = args.start_epoch

    save_dir = args.save_dir

    config, net, loss, get_pbb = get_model(args.model, args)

    if args.resume:

        checkpoint = torch.load(args.resume)

        if start_epoch == 0:

            start_epoch = checkpoint['epoch'] + 1

        if not save_dir:

            save_dir = checkpoint['save_dir']

        else:

            save_dir = os.path.join('results',save_dir)

        net = load_and_save_net_model.netio_load(net, checkpoint['state_dict'], False)

    else:

        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            year = str(datetime.datetime.now().year)
            month = str(datetime.datetime.now().month)
            day   = str(datetime.datetime.now().day)
            exp_id = year+month+day
            save_dir = os.path.join('results', args.model + '-' + exp_id)
        else:
            save_dir = os.path.join('results',save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logfile = os.path.join(save_dir, 'log')

    if args.copy_file == True:
        sys.stdout = message_logger.Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f,os.path.join(save_dir,f))

    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True


    if args.train_mode == "train":

        train_dataset = ClassifierDataSet(args, args.preprocess_path_train, args.train_splite_txt_path, "train")

        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True)

        val_dataset = ClassifierDataSet(args, args.preprocess_path_valid, args.validate_splite_txt_path, "val")

        val_loader = DataLoader( val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4, pin_memory=True, drop_last=True)

    else:

        val_dataset = ClassifierDataSet(args, args.preprocess_path_test, args.test_splite_txt_path, "val")

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4, pin_memory=True)

    wraper = Wraper(net, loss)

    wraper = DataParallel(wraper, device_ids=args.gpu)

    trainer = Trainer(wraper, args)

    if args.train_mode == "train":

        trainer.do_train(args, train_loader, val_loader, save_dir, start_epoch)

    else:

        trainer.do_test(args, val_loader)








if __name__ == '__main__':
    main()