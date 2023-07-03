import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from random import randint, sample

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def fileread(data_dir):
    print(data_dir)
    with open(data_dir, 'rb') as f:
        reader = f.readlines()
        data = []
        for line in reader:
            data.append(line.decode().strip().split(','))
    # print(len(data))
    data = np.array(data, dtype=float)
    return data  # .reshape([len(data),4,1])


def read_csv(data_path):
    print("reading csv : %s" % (data_path))
    use_cols = [line.strip() for line in open('config/used_header_info')]
    print('used_header_info 使用特征{}个'.format(len(use_cols)))

    # get dtype
    dtype = dict()
    for name in use_cols:
        if 'CATEGORY' in name:
            print(name)
            dtype[name] = 'category'
    if dtype:
        print('使用特征指定数据类型::', dtype)

    df = pd.read_csv(data_path,
                     header=0, usecols=use_cols, dtype=dtype, keep_default_na=False, nrows=500000)
    # print(df.dtypes)
    return df


def select_user_data(df):
    use_cols = [line.strip() for line in open('config/user_feature_info')]
    print('user_feature_info 使用特征{}个'.format(len(use_cols)))
    return df[use_cols].to_numpy(dtype='float32')


def select_item_data(df):
    use_cols = [line.strip() for line in open('config/product_feature_info')]
    print('product_feature_info 使用特征{}个'.format(len(use_cols)))
    # 待修改为item_id, torch.float对应np.float32, np默认np.float64,对应torch.double
    return df[use_cols].to_numpy(dtype='float32')


def fileread_propensity(data_dir):
    print(data_dir)
    with open(data_dir, 'rb') as f:
        reader = f.readlines()
        data = []
        for line in reader:
            data.append(line.decode().strip().split(' '))
    # print(len(data))
    data = np.array(data, dtype=float)
    return data  # .reshape([len(data),4,1])


# class DataLoad_huawei(data.Dataset):
#     def __init__(self, root, data_feature_path=None, uni_dataset_dir=None, uni_percent=1.0, syn=False):
#         # self.user_item = fileread(os.path.join(root,'dataset'))
#         # self.user_item = fileread(root)

#         # self.user_item = np.abs(np.random.rand(10000, 600))

#         self.data_df = read_csv(root)
#         self.user_item = select_user_data(self.data_df)
#         self.item_feature = select_item_data(self.data_df)
#         self.label = self.data_df['click'].to_numpy()
#         self.impression = self.data_df['impression'].to_numpy()

#         if self.user_item.size == 0:
#             print('user_item is empty')
#         if self.item_feature.size == 0:
#             print('item_feature is empty')
#         if self.label.size == 0:
#             print('label is empty')
#         if self.impression.size == 0:
#             print('impression is empty')

#         self.transforms = transforms.Compose([transforms.ToTensor()])
#         self.uni_dataset_dir = uni_dataset_dir
#         # self.data_feature_path = data_feature_path

#     def __getitem__(self, idx):
#         # print(idx) 300dim x

#         user_fea = torch.from_numpy(self.user_item[idx])
#         item_fea = torch.tensor(self.item_feature[idx])
#         impression = torch.tensor([self.impression[idx]])
#         click = torch.tensor([self.label[idx]])
#         return user_fea, item_fea, click, impression

#         # data = torch.from_numpy(self.user_item[idx])
#         # impression = torch.from_numpy(np.ones(1))
#         # if idx % 2 == 1:
#         #     item = torch.from_numpy(np.ones(1))
#         # else:
#         #     item = torch.from_numpy(np.zeros(1))
#         #
#         # # numerical_feature, item_id, label, impression
#         # return data[4:304], item, item, impression


#     def __len__(self):
#         return len(self.user_item)


class DataLoad_huawei(data.Dataset):
    def __init__(self, root, mode):
        # self.user_item = fileread(os.path.join(root,'dataset'))
        # self.user_item = fileread(root)

        # self.user_item = np.abs(np.random.rand(10000, 600))

        self.data_df = pd.read_csv(root, header=0, nrows=100000).values
        self.mode = mode
        partition = int(0.75 * self.data_df.shape[0])
        if self.mode == 'dev':
            self.data_df = self.data_df[:partition]
        elif self.mode == 'test':
            self.data_df = self.data_df[partition:]

        self.transforms = transforms.Compose([transforms.ToTensor()])
        # self.data_feature_path = data_feature_path

    def __getitem__(self, idx):
        # print(idx) 300dim x
        data = torch.from_numpy(self.data_df[idx])
        # user_fea = torch.from_numpy(self.data_df[idx, : -2])
        # item_fea = torch.from_numpy(self.data_df[idx: 42: 153])
        # match = torch.from_numpy(self.data_df[idx, 153:-2])
        # impression = torch.tensor(self.data_df[idx, -2])
        # click = torch.tensor(self.data_df[idx, -1])
        # print(self.data_df[idx])
        return data[253:282], data[42: 253], data[-1], data[-2]

        # data = torch.from_numpy(self.user_item[idx])
        # impression = torch.from_numpy(np.ones(1))
        # if idx % 2 == 1:
        #     item = torch.from_numpy(np.ones(1))
        # else:
        #     item = torch.from_numpy(np.zeros(1))
        #
        # # numerical_feature, item_id, label, impression
        # return data[4:304], item, item, impression

    def __len__(self):
        return self.data_df.shape[0]


def dataload_huawei(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, mode='dev'):
    if batch_size == 'all':
        if mode == 'dev':
            batch_size = 75000
        else:
            batch_size = 25000
    dataset = DataLoad_huawei(dataset_dir, mode)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return dataset


class DataLoad_ori(data.Dataset):
    def __init__(self, root, data_feature_path=None, uni_dataset_dir=None, uni_percent=1.0, syn=False,
                 ipm_pair_path=None, ipm_sample_path=None, mode='test', nagetive_path=None, fairness=False):
        # self.user_item = fileread(os.path.join(root,'data'))
        self.user_item = fileread(root)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.uni_dataset_dir = uni_dataset_dir
        self.data_feature_path = data_feature_path
        self.negative_sample = False
        self.mode = mode
        self.fairness = fairness
        if nagetive_path is not None:
            self.negative = np.load(nagetive_path)
            self.negative_sample = True
        if self.uni_dataset_dir is not None:
            self.user_item_uniform = fileread(uni_dataset_dir)
            n = int(len(self.user_item_uniform) * uni_percent)
            # shuffle
            random_rows = torch.from_numpy(
                np.random.randint(0, len(self.user_item_uniform), size=len(self.user_item_uniform)))
            self.user_item_uniform = self.user_item_uniform[random_rows, :]
            # print(self.user_item_uniform.shape)
            self.len_user_item_uniform = n

        if data_feature_path is not None:
            if syn:
                self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'),
                                           allow_pickle=True).item()
                # print(self.userfeature)
                self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'),
                                           allow_pickle=True).item()
            else:
                self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'),
                                           allow_pickle=True)  # .item()
                self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'),
                                           allow_pickle=True)  # .item()
            # print(self.userfeature)

        if ipm_sample_path is not None:
            # self.ipm_pair = np.load(ipm_pair_path, allow_pickle=True).item()
            self.ipm_sample = np.load(ipm_sample_path)
            self.sample = True
        else:
            self.sample = False
        if mode == 'test':
            n = int(len(self.user_item) * 0)
            # shuffle
            self.user_item = self.user_item

    # def sample_ipm_data(self):
    #     a1, a2 = np.random.choice(np.array(list(self.ipm_pair.keys())), size=len(self.user_item), replace=True,
    #                               p=np.array(list(self.ipm_pair.values())))[0].strip().split('_')
    #     self.ipm_data1 =

    # def sample_ipm_pair(self, idx):
    #     np.random.seed(idx)
    #     a1, a2 = np.random.choice(np.array(list(self.ipm_pair.keys())), size=1, replace=True, p=np.array(list(self.ipm_pair.values())))[0].strip().split('_')
    #     # print(a1, a2)
    #     # print(self.ipm_sample[int(a1)], a2)
    #     #
    #     # idx1 = np.random.choice(np.arange(len(self.ipm_sample[int(a1)])), size=1, replace=True)
    #     # idx2 = np.random.choice(np.arange(len(self.ipm_sample[int(a2)])), size=1, replace=True)
    #     len1 = len(self.ipm_sample[int(a1)])
    #     len2 = len(self.ipm_sample[int(a2)])
    #     # print(self.ipm_sample[int(a1)][idx % len1])
    #     return self.ipm_sample[int(a1)][idx % len1], self.ipm_sample[int(a2)][idx % len2]

    def __getitem__(self, idx):
        # print(idx)
        data = torch.from_numpy(self.user_item[idx])
        if self.uni_dataset_dir is not None:
            data_uniform = torch.from_numpy(self.user_item_uniform[idx % self.len_user_item_uniform])

        if self.negative_sample:
            self.negative_data = torch.from_numpy(self.negative[idx]).int()

        if self.data_feature_path is not None:
            if self.uni_dataset_dir is not None:

                return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                  torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), \
                       torch.from_numpy(self.itemfeature[int(data[1].numpy())]), \
                       data[2].int(), data[3].int(), \
                       torch.cat((torch.from_numpy(self.userfeature[int(data_uniform[0].numpy())]),
                                  torch.from_numpy(self.itemfeature[int(data_uniform[1].numpy())])), 0), \
                       torch.from_numpy(self.itemfeature[int(data_uniform[1].numpy())]), \
                       data_uniform[2].int(), data_uniform[3].int()
            else:

                if self.sample:
                    data1, data2 = torch.from_numpy(self.ipm_sample[idx % len(self.ipm_sample)][:4]), torch.from_numpy(
                        self.ipm_sample[idx % len(self.ipm_sample)][4:])
                    # print(data[0])
                    return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), \
                           torch.from_numpy(self.itemfeature[int(data[1].numpy())]), \
                           data[2].int(), data[3].int(), \
                           torch.cat((torch.from_numpy(self.userfeature[int(data1[0])]),
                                      torch.from_numpy(self.itemfeature[int(data1[1].numpy())])), 0), \
                           torch.from_numpy(self.itemfeature[int(data1[1].numpy())]), \
                           data1[2].int(), data1[3].int(), \
                           torch.cat((torch.from_numpy(self.userfeature[int(data2[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data2[1].numpy())])), 0), \
                           torch.from_numpy(self.itemfeature[int(data2[1].numpy())]), \
                           data2[2].int(), data2[3].int()
                if self.mode == 'test':
                    if self.fairness:
                        return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), \
                           torch.from_numpy(self.itemfeature[int(data[1].numpy())]), \
                           data[2].int(), data[3].int(), data[4].int(), data[5].int()
                    else:
                        return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), \
                           torch.from_numpy(self.itemfeature[int(data[1].numpy())]), \
                           data[2].int(), data[3].int()
                else:
                    if self.fairness:
                        return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), \
                           torch.from_numpy(self.itemfeature[int(data[1].numpy())]), \
                           data[2].int(), data[3].int(), data[4].int(), data[5].int()
                    else:
                        return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), \
                           torch.from_numpy(self.itemfeature[int(data[1].numpy())]), \
                           data[2].int(), data[3].int()

        else:
            if self.uni_dataset_dir is not None:
                # print(data_uniform[0], data_uniform[1], data_uniform[2], data_uniform[3])
                return data[0], data[1], data[2], data[3], data_uniform[0], data_uniform[1], data_uniform[2], \
                       data_uniform[3]
            else:

                if self.sample:
                    # print(self.ipm_sample[idx])
                    # print(self.ipm_sample.shape)
                    data1, data2 = torch.from_numpy(self.ipm_sample[idx % len(self.ipm_sample)][:4]), torch.from_numpy(
                        self.ipm_sample[idx % len(self.ipm_sample)][4:])
                    # print(data1, data2)
                    return data[0], data[1], data[2], data[3], data1[0], data1[1], data1[2], data1[3], data2[0], data2[
                        1], data2[2], data2[3]
                if self.mode == 'test' and self.negative_sample:
                    return data[0], data[1], data[2], data[3], self.negative_data
                else:
                    return data[0], data[1], data[2], data[3]

    def __len__(self):
        # print(len(self.user_item))
        return len(self.user_item)


def dataload_ori(dataset_dir, batch_size, shuffle=True, num_workers=8, pin_memory=True, data_feature_path=None,
                 uniform_dir=None, uni_percent=1.0, syn=False, ipm_pair_path=None, ipm_sample_path=None, mode='train', nagetive_path=None, fairness=False):

    dataset = DataLoad_ori(dataset_dir, data_feature_path=data_feature_path, uni_dataset_dir=uniform_dir,
                           uni_percent=uni_percent, syn=syn, ipm_pair_path=ipm_pair_path,
                           ipm_sample_path=ipm_sample_path, mode=mode, nagetive_path=nagetive_path, fairness=fairness)

    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    return dataset


class DataLoad(data.Dataset):
    def __init__(self, root, propensity=None, imputation=None, data_feature_path=None, a_propensity=False, Flag=False, mode='train', nagetive_path=None):
        # self.user_item = fileread(os.path.join(root,'data'))
        self.user_item = fileread(root)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.propensity = propensity
        self.imputation = imputation
        # print('xxxxxxxxxxxxxxxxxxxx', data_feature_path)
        self.data_feature_path = data_feature_path
        self.a_propensity = a_propensity
        self.mode = mode
        self.negative_sample = False
        if nagetive_path is not None:
            self.negative = np.load(nagetive_path)
            self.negative_sample = True
        '''
        if data_feature_path is not None:
            self.propensity_list = fileread_propensity(propensity)
        '''
        if imputation is not None:
            self.imputation_list = np.c_[self.user_item, fileread(imputation)]
            self.len_imputation_list = len(self.imputation_list)
        if data_feature_path is not None and not Flag:
            self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'),
                                       allow_pickle=True).item()
            self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'),
                                       allow_pickle=True).item()
        elif data_feature_path is not None:
            self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'),
                                       allow_pickle=True)
            self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'),
                                       allow_pickle=True)

    def __getitem__(self, idx):
        # print(idx)
        '''
        if self.imputation is not None:#%min(len(self.imputation_list), len(self.user_item))
            imputation_data = torch.from_numpy(self.imputation_list[idx])
            if self.data_feature_path is not None:
                #print('xxxxxxxxxxxxxxxxxxxx')
                imputation_data[0], imputation_data[1], imputation_data[4], imputation_data[5] = torch.from_numpy(self.userfeature[int(imputation_data[0].numpy())]), torch.from_numpy(self.userfeature[int(imputation_data[1].numpy())]), torch.from_numpy(self.userfeature[int(imputation_data[4].numpy())]), torch.from_numpy(self.userfeature[int(imputation_data[5].numpy())])
            #print(imputation_data)
        '''

#         print('xxxxxxxxxxxxxxxxxxxx', self.data_feature_path)
        data = torch.from_numpy(self.user_item[idx])
#         self.negative_data=None
        if self.negative_sample:
            self.negative_data = torch.from_numpy(self.negative[idx])
#

        if self.propensity is not None:
            propensity_data = torch.from_numpy(int(self.propensity_list[data[0]]), int(self.propensity_list[data[1]]))
            '''
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[4], imputation_data[5], imputation_data[6], imputation_data[7], propensity_data
            '''
            if self.data_feature_path is not None:
#                 print('xxxxxxxxxxxxxxxxxxxx')
                return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), torch.from_numpy(
                    self.itemfeature[int(data[1].numpy())]), data[2], data[3], propensity_data
            if self.mode == 'train':
                return data[0], data[1], data[2], data[3], propensity_data
            else:
                return data[0], data[1], data[2], data[3], propensity_data, self.negative_data
        # print(data[0])
        else:
            '''
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[4], imputation_data[5], imputation_data[6], imputation_data[7]
            '''
            # print(data[0])
            if self.data_feature_path is not None:
#                 print('xxxxxxxxxxxxxxxxxxxx')
                # print(data[0], data[1])
                return torch.cat((torch.from_numpy(self.userfeature[int(data[0].numpy())]),
                                      torch.from_numpy(self.itemfeature[int(data[1].numpy())])), 0), torch.from_numpy(
                    self.itemfeature[int(data[1].numpy())]), data[2], data[3], data[1]
            if self.mode == 'train' or not self.negative_sample:
                return data[0], data[1], data[2], data[3]
            else:
                return data[0], data[1], data[2], data[3], self.negative_data

    def __len__(self):
        if self.imputation is not None:
            # print(len(self.imputation_list), len(self.user_item))
            return len(self.imputation_list)
        return len(self.user_item)


def dataload(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, propensity=None, imputation=None,
             data_feature_path=None, a_propensity=False, Flag=False, mode='train', nagetive_path=None):
    dataset = DataLoad(dataset_dir, propensity, imputation, data_feature_path, a_propensity, Flag, mode=mode, nagetive_path=nagetive_path)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    return dataset


class DataLoadPretrain(data.Dataset):
    def __init__(self, root, propensity=None, imputation=None, data_feature_path=None, pretrain_mode='propensity', syn=False):
        # self.user_item = fileread(os.path.join(root,'data'))
        self.user_item = fileread(root)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.propensity = propensity
        self.imputation = imputation
        self.pretrain_mode = pretrain_mode
        # print('xxxxxxxxxxxxxxxxxxxx', data_feature_path)
        self.data_feature_path = data_feature_path
        if propensity is not None:
            self.propensity_list = fileread_propensity(propensity)
        if imputation is not None:
            self.imputation_list = np.c_[self.user_item, fileread(imputation)]
            self.len_imputation_list = len(self.imputation_list)
        if data_feature_path is not None and not syn:
            self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'),
                                       allow_pickle=True)
            self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'),
                                       allow_pickle=True)
        elif data_feature_path is not None and syn:
            self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'),
                                       allow_pickle=True).item()
            self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'),
                                       allow_pickle=True).item()

    def __getitem__(self, idx):
        # print(idx)
        '''
        if self.imputation is not None:#%min(len(self.imputation_list), len(self.user_item))
            imputation_data = torch.from_numpy(self.imputation_list[idx])
            if self.data_feature_path is not None:
                #print('xxxxxxxxxxxxxxxxxxxx')
                imputation_data[0], imputation_data[1], imputation_data[4], imputation_data[5] = torch.from_numpy(self.userfeature[int(imputation_data[0].numpy())]), torch.from_numpy(self.userfeature[int(imputation_data[1].numpy())]), torch.from_numpy(self.userfeature[int(imputation_data[4].numpy())]), torch.from_numpy(self.userfeature[int(imputation_data[5].numpy())])
            #print(imputation_data)
        '''

        # print('xxxxxxxxxxxxxxxxxxxx', self.data_feature_path)
        data = torch.from_numpy(self.user_item[idx])

        if self.propensity is not None:

            propensity_data = torch.from_numpy(int(self.propensity_list[data[0]]), int(self.propensity_list[data[1]]))
            '''
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[4], imputation_data[5], imputation_data[6], imputation_data[7], propensity_data
            '''
            if self.data_feature_path is not None:
                # print('xxxxxxxxxxxxxxxxxxxx')
                if self.pretrain_mode == 'propensity':
                    return torch.from_numpy(np.concatenate((self.userfeature[int(data[0].numpy())], self.itemfeature[int(data[1].numpy())]), axis=0)), data[1], data[2], data[
                        3], propensity_data
                return torch.from_numpy(np.concatenate((self.userfeature[int(data[0].numpy())], self.itemfeature[int(data[1].numpy())]), axis=0)), torch.from_numpy(
                    self.itemfeature[int(data[1].numpy())]), data[2], data[3], propensity_data
            return data[0], data[1], data[2], data[3], propensity_data
        # print(data[0])
        else:
            '''
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[4], imputation_data[5], imputation_data[6], imputation_data[7]
            '''
            # print('xxxxxxxxxxxxxxxxxxxx')
            # print(data[0])
            if self.data_feature_path is not None:
                # print('xxxxxxxxxxxxxxxxxxxx')
                # print(data[0], data[1])
                if self.pretrain_mode == 'propensity':
                    return torch.from_numpy(np.concatenate((self.userfeature[int(data[0].numpy())], self.itemfeature[int(data[1].numpy())]), axis=0)), data[1], data[2], data[3]
                return torch.from_numpy(np.concatenate((self.userfeature[int(data[0].numpy())], self.itemfeature[int(data[1].numpy())]), axis=0)), torch.from_numpy(
                    self.itemfeature[int(data[1].numpy())]), data[2], data[3]
            return data[0], data[1], data[2], data[3]

    def __len__(self):
        if self.imputation is not None:
            # print(len(self.imputation_list), len(self.user_item))
            return len(self.imputation_list)
        return len(self.user_item)


class DataLoad_Sample(data.Dataset):
    def __init__(self, root, propensity=None, imputation=None, data_feature_path=None, imputation_model=None):
        # self.user_item = fileread(os.path.join(root,'data'))
        self.user_item = fileread(root)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.propensity = propensity
        self.imputation = imputation
        self.imputation_model = imputation_model
        self.data_feature_path = data_feature_path
        if propensity is not None:
            self.propensity_list = fileread_propensity(propensity)
        if imputation_model is not None:
            self.full_data = np.load(os.path.join('./', 'dataset', 'yahoo', 'full_data.txt.npy'))
            print(self.full_data.shape)
        if data_feature_path is not None:
            self.userfeature = np.load(os.path.join(data_feature_path, 'user_features.npy'), allow_pickle=True)
            self.itemfeature = np.load(os.path.join(data_feature_path, 'item_features.npy'), allow_pickle=True)

    def __getitem__(self, idx):
        # print(idx)
        if self.imputation_model is not None:  # %min(len(self.imputation_list), len(self.user_item))
            x = np.random.randint(0, 15400, size=1, dtype='l')
            a = np.random.randint(0, 1000, size=1, dtype='l')
            imputation_data = torch.argmax(self.imputation_model.predict(torch.tensor([int(x)]).to(device),
                                                                         torch.tensor([int(a)]).to(
                                                                             device))).detach().numpy() if \
            self.full_data[x, a] == -1 else self.full_data[x, a]
            if self.data_feature_path is not None:
                imputation_data[0], imputation_data[1], imputation_data[4], imputation_data[5] = torch.from_numpy(
                    self.userfeature[int(imputation_data[0].numpy())]), torch.from_numpy(
                    self.userfeature[int(imputation_data[1].numpy())]), torch.from_numpy(
                    self.userfeature[int(imputation_data[4].numpy())]), torch.from_numpy(
                    self.userfeature[int(imputation_data[5].numpy())])
            # print(imputation_data)
        else:
            data = torch.from_numpy(self.user_item[idx])
            if self.data_feature_path is not None:
                data[0], data[1] = torch.from_numpy(self.userfeature[int(data[0].numpy())]), torch.from_numpy(
                    self.userfeature[int(data[1].numpy())])

        if self.propensity is not None:
            propensity_data = torch.from_numpy(int(self.propensity_list[data[0]]), int(self.propensity_list[data[1]]))
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[
                    4], imputation_data[5], imputation_data[6], imputation_data[7], propensity_data
            return data[0], data[1], data[2], data[3], propensity_data
        # print(data[0])
        else:
            if self.imputation is not None:
                return imputation_data[0], imputation_data[1], imputation_data[2], imputation_data[3], imputation_data[
                    4], imputation_data[5], imputation_data[6], imputation_data[7]
            return data[0], data[1], data[2], data[3]

    def __len__(self):
        return len(self.user_item)


def dataload_sample(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, propensity=None,
                    imputation=None, data_feature_path=None, imputation_model=None):
    dataset = DataLoad_Sample(dataset_dir, propensity, imputation, data_feature_path, imputation_model)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return dataset


def pretrain_dataload(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True, propensity=None,
                      imputation=None, data_feature_path=None, pretrain_mode='propensity', syn=False):
    dataset = DataLoadPretrain(dataset_dir, propensity, imputation, data_feature_path, pretrain_mode, syn=syn)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return dataset


if __name__ == '__main__':
    dataload_huawei('./dataset/part_0_test')
