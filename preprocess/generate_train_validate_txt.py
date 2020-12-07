import argparse
import os
import numpy as np
import random
import glob



parser = argparse.ArgumentParser(description='PyTorch Generate Classifier Train and Validate Txt')


parser.add_argument('--train_ratio', default=0.0 , type=float, help= "train/validate ratio")

parser.add_argument('--preprocess_path_train', default="/data_ssd/lxw/LungNodule/mvd_0902_classifier_small_crop_image_validate/",type=str, help="train data preprocessed path")
parser.add_argument('--train_splite_txt_path',    default="/data_ssd/lxw/LungNodule/mvd_0902_classifier_small_crop_image_validate/train_splite.txt",type=str, help="train splite txt")
parser.add_argument('--validate_splite_txt_path', default="/data_ssd/lxw/LungNodule/mvd_0902_classifier_small_crop_image_validate/validate_splite.txt",type=str, help="valid splite txt")
parser.add_argument('--white_list_path', default="",  type=str, help = "train splite name")



if __name__ == '__main__':

    args = parser.parse_args()

    original_data_path = args.preprocess_path_train

    all_image_list = glob.glob(os.path.join(original_data_path, "*_image_mask_*.npy"))

    random.shuffle(all_image_list)

    train_sample_number     = int(args.train_ratio * len(all_image_list))

    validate_sample_number  = len(all_image_list) - train_sample_number

    train_set = all_image_list[0:train_sample_number]
    valit_set = all_image_list[train_sample_number::]

    # exit(0)

    white_case_id = []

    if args.white_list_path:

        white_list_path = args.white_list_path

        with open(white_list_path, "r") as f:
            current_white_case_id = f.read()

        white_case_id = current_white_case_id.split("\n")


    train_saved_txt = args.train_splite_txt_path
    valid_saved_txt = args.validate_splite_txt_path

    for train_path in train_set:

        current_number = train_path.split("/")[-1]
        current_label  = current_number.split("_")[-1].split(".")[0]
        current_recoard = current_number + " " + current_label

        if current_number in white_case_id:
            continue

        with open(train_saved_txt, "a+") as write_txt:

            write_txt.write(current_recoard+ "\n")

    for valid_path in valit_set:

        current_number = valid_path.split("/")[-1]
        current_label  = current_number.split("_")[-1].split(".")[0]
        current_recoard = current_number + " " + current_label

        if current_number in white_case_id:
            continue

        with open(valid_saved_txt, "a+") as write_txt:

            write_txt.write(current_recoard+ "\n")