# train_path = "./../../data/Synapse/train_npz"
# train_list = "./../lists/lists_Synapse"
train_list = r"D:\2_study\6_CC_code\TransUNet_CC\down\TransUNet\lists\list_lits"
import os
import random


def label_data_index_process(train_path, dataset, train=True, ratio=0.2):
    label_index = []
    unlabel_index = []
    if train:
        file_list = open(os.path.join(train_path, "train.txt")).readlines()
        file_dict = {}
        for index, item in enumerate(file_list):
            if dataset == "Synapse":
                cur_case = item[4:8]
            elif dataset == "ACDC":
                cur_case = item[5:17]
            elif dataset == "LITS":
                cur_case = item[:3]
            else:
                cur_case = item[5:8] # LA
            if cur_case not in file_dict.keys():
                tmp_list = [index]
                file_dict[cur_case] = tmp_list
            else:
                file_dict[cur_case].append(index)
        for key in file_dict.keys():
            length = len(file_dict[key])
            if dataset == "Synapse":
                file_dict_key = file_dict[key]
                random.seed(12)
                random.shuffle(file_dict_key)
                label_index += file_dict_key[:int(length * ratio)]
                unlabel_index += file_dict_key[int(length * ratio):]
            elif dataset == "LA":
                file_dict_key = file_dict[key]
                random.seed(12)
                random.shuffle(file_dict_key)
                label_index += file_dict_key[:int(length * ratio)]
                unlabel_index += file_dict_key[int(length * ratio):]
            else:
                label_index += file_dict[key][:int(length*ratio)]
                unlabel_index += file_dict[key][int(length*ratio):]
    return label_index, unlabel_index
#
# label_index, unlabel_index = label_data_index_process(train_path= train_list, dataset="LITS")
# print(label_index)
# print(unlabel_index)

# train_path = r"D:\2_study\6_CC_code\TransUNet_CC\down\TransUNet\lists\list_lits\train.txt"
