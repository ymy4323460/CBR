import os
import math
import time
import random
import torch
import argparse
import numpy as np
import torch
import torch.nn as nn
import utils as ut
import torch.optim as optim
from torch import autograd
from torch.utils import data
# import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
import dataloader as dataload
from codebase.adversarial.config import get_config
from codebase.adversarial.learner import Learner
import sklearn.metrics as skm
import warnings
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print("torch.cuda.is_available():%s" % (torch.cuda.is_available()))

args, _ = get_config()
workstation_path = './'
# train_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'data_nonuniform.csv')
# test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'data_nonuniform.csv')

if args.dataset == 'huawei':
    train_dataset_dir = os.path.join(workstation_path, 'dataset/', 'huawei', 'dev.csv')
    test_dataset_dir = os.path.join(workstation_path, 'dataset/', 'huawei', 'dev.csv')
    pretrain_dataloader = dataload.dataload_huawei(train_dataset_dir, args.batch_size, mode='dev')
    if args.impression_or_click == 'impression':
        train_dataloader = dataload.dataload_huawei(train_dataset_dir, args.batch_size, mode='dev')
    test_dataloader = dataload.dataload_huawei(test_dataset_dir, args.batch_size, mode='test')
else:
    if args.dataset[:9] == 'logitdata' or args.dataset[:7] == 'invdata' or args.dataset[:3] == 'non':
        syn = True
        train_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'data_nonuniform.csv')
        test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'data_uniform.csv')
    else:
        syn = False
        train_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'data_nonuniform.csv')
        test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'data_uniform.csv')


    nagetive_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'negative_samples.npy')

    if args.debias_mode == 'IPM_Embedding_Sample':
        # ipm_pair_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'pair_dict.npy')
        # ipm_sample_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'sample_dict.npy')
        ipm_pair_path = None
        ipm_sample_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'ipm_data',
                                       'ipm_data5_1.npy')

    else:
        ipm_pair_path = None
        ipm_sample_path = None

    if args.feature_data:
        data_feature_path = os.path.join(workstation_path, 'dataset', args.dataset)
    else:
        data_feature_path = None
    pretrain_dataloader = dataload.dataload_ori(train_dataset_dir, args.batch_size, data_feature_path=data_feature_path,
                                                syn=syn)
    if args.debias_mode == 'IPM_Embedding_K':
        ipm_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'ipm_data', 'data_ipm5.npy')
        ipm_dataloader = dataload.dataload_ipm(ipm_dataset_dir, args.batch_size, data_feature_path=data_feature_path,
                                               syn=syn)

    if args.impression_or_click == 'impression':
        train_dataloader = dataload.dataload_ori(train_dataset_dir, args.batch_size,
                                                 data_feature_path=data_feature_path, syn=syn,
                                                 ipm_sample_path=ipm_sample_path)
    test_dataloader = dataload.dataload_ori(test_dataset_dir, 1, data_feature_path=data_feature_path,
                                            syn=syn, mode='test', nagetive_path=nagetive_path)
    test_dataloader_auc_acc = dataload.dataload_ori(test_dataset_dir, args.batch_size, data_feature_path=data_feature_path,
                                            syn=syn, mode='test', nagetive_path=nagetive_path)
    val_dataloader = dataload.dataload_ori(test_dataset_dir, args.batch_size, data_feature_path=data_feature_path,
                                           syn=syn, mode='val')

# 构建模型
model = Learner(args)
logistic = torch.nn.Sigmoid()
if args.debias_mode in ['Adversarial']:
    # training model
    history = np.zeros((args.epoch_max, 3))
    for i in model.parameters():
        i.requires_grad = False
    for i in model.debias_model.discriminitor.parameters():
        i.requires_grad = True
    # define optimizer for max
    max_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.999))

    for i in model.parameters():
        i.requires_grad = True
    for i in model.debias_model.discriminitor.parameters():
        i.requires_grad = False
    # define optimizer for min
    min_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.999))

    for epoch in range(args.epoch_max):
        model.train()
        emb_loss_max = 0
        emb_loss_min = 0
        ctr_loss = 0
        total_emb_loss = 0
        total_ctr_loss = 0
        auc = 0
        total_auc = 0
        total_acc = 0
        total_logloss = 0
        total_confounder_loss = 0
        # for step in range(args.max_step):
        # print("self.args.impression_or_click:%s" % args.impression_or_click)
        if args.impression_or_click == 'impression':
            #  numerical_feature, item_id, label, impression
            for x, a, y, r in train_dataloader:
                max_optimizer.zero_grad()
                x, a, y, r = x.to(device), a.to(device), y.to(device), r.to(device)
                # print(x.size())
                emb_loss_max = -model.learn(x, a, r, y, savestep=epoch, minmax_turn='max')
                emb_loss_max.backward()
                max_optimizer.step()
                total_emb_loss += emb_loss_max.item()
                test_dataloader_len = len(train_dataloader)

                min_optimizer.zero_grad()
                # min_optimizer2.zero_grad()
                if args.confounder:
                    # print(x.size(), a.size(), r.size(), y.size())
                    emb_loss_min, ctr_loss, confounder_loss = model.learn(x, a, r, y, epoch, minmax_turn='min')
                    # print(ctr_loss)
                    total_confounder_loss += confounder_loss.item()
                else:
                    if args.is_debias:
                        emb_loss_min, ctr_loss = model.learn(x, a, r, y, epoch, minmax_turn='min')
                        # print(ctr_loss)
                    else:
                        ctr_loss = model.learn(x, a, r, y, epoch, minmax_turn='min')
                if args.downstream == 'MLP':
                    y_pred = model.predict(x, a)
                    auc = skm.roc_auc_score(y.cpu().numpy(), y_pred.cpu().detach().numpy())
                    acc = skm.accuracy_score(y.cpu().numpy(), torch.where(y_pred > 0.5, torch.ones_like(y_pred),
                                                                          torch.zeros_like(
                                                                              y_pred)).cpu().detach().numpy())
                    logloss = skm.log_loss(y.cpu().numpy(), y_pred.cpu().detach().numpy())
                else:
                    y_pred = model.predict(x, a)
                    auc = skm.roc_auc_score(y.cpu().numpy(), y_pred.cpu().detach().numpy())
                    acc = skm.accuracy_score(y.cpu().numpy(), torch.where(y_pred > 0.5, torch.ones_like(y_pred),
                                                                          torch.zeros_like(
                                                                              y_pred)).cpu().detach().numpy())
                    logloss = skm.log_loss(y.cpu().numpy(), y_pred.cpu().detach().numpy())

                L1 = emb_loss_min + ctr_loss if args.is_debias else ctr_loss
                #
                # print(L.size())
                L1.backward()
                min_optimizer.step()

                # min_optimizer2.step()
                total_ctr_loss += ctr_loss.item()
                total_auc += auc
                total_acc += acc
                total_logloss += logloss

        if epoch % 1 == 0:
            total_test_auc = 0
            total_test_acc = 0
            total_test_logloss = 0
            total_test_ndcg = 0
            total_test_recall = 0
            total_test_precision = 0
            test_dataloader_len = len(test_dataloader)
            test_dataloader_auc_acc_len = len(test_dataloader_auc_acc)
            train_dataloader_len = len(train_dataloader)
            for test_x, test_a, test_y, test_r, negative_a in test_dataloader:
#                 print(test_a)
                negative_a = negative_a.reshape(-1)
                if test_y == 0:
                    test_dataloader_len -= 1
                    continue


                if args.downstream == 'MLP':
                    test_a = torch.cat([test_a, negative_a], axis=0)
                    test_x = test_x.repeat(100)
                    test_y_list = torch.zeros(100)
                    test_y_list[0] = test_y
                    test_y = test_y_list
                    test_y_pred = F.sigmoid(model.predict(test_x, test_a)).cpu().detach().numpy().reshape(-1)
                    top20 = np.sort(test_y_pred)[9]
#                     test_acc = skm.accuracy_score(test_y.cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
#                                                                           np.zeros_like(test_y_pred)))
                    test_pre = skm.precision_score(test_y.cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
                                                                          np.zeros_like(test_y_pred)))
                    test_rec = skm.recall_score(test_y.cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
                                                                          np.zeros_like(test_y_pred)))
                    test_ndcg = skm.ndcg_score(np.expand_dims(test_y.cpu().numpy(), axis=0), np.expand_dims(test_y_pred, axis=0), k=10)

                else:
#                     test_y_pred = model.predict(test_x, test_a).cpu().detach().numpy()
                    test_a = torch.cat([test_a, negative_a], axis=0)
                    test_x = test_x.repeat(100)
                    test_y_list = torch.zeros(100)
                    test_y_list[0] = test_y
                    test_y = test_y_list
                    test_y_pred = F.sigmoid(model.predict(test_x, test_a)).cpu().detach().numpy()
                    top20 = np.sort(test_y_pred)[9]
#                     test_acc = skm.accuracy_score(test_y.cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
#                                                                           np.zeros_like(test_y_pred)))
                    test_pre = skm.precision_score(test_y.cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
                                                                          np.zeros_like(test_y_pred)))
                    test_rec = skm.recall_score(test_y.cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
                                                                          np.zeros_like(test_y_pred)))
                    test_ndcg = skm.ndcg_score(np.expand_dims(test_y.cpu().numpy(), axis=0), np.expand_dims(test_y_pred, axis=0), k=10)

                total_test_ndcg += test_ndcg
                total_test_recall += test_rec
                total_test_precision += test_pre

            for test_x, test_a, test_y, test_r, negative_a in test_dataloader_auc_acc:

                if args.downstream == 'MLP':
                    test_y_pred = F.sigmoid(model.predict(test_x, test_a)).cpu().detach().numpy()
                    test_auc = skm.roc_auc_score(test_y.cpu().numpy(), test_y_pred)
                    test_acc = skm.accuracy_score(test_y.cpu().numpy(),
                                                  np.where(test_y_pred > 0.5, np.ones_like(test_y_pred),
                                                              np.zeros_like(test_y_pred)))
                    test_logloss = skm.log_loss(test_y.cpu().detach().numpy(), test_y_pred)
                else:
                    test_y_pred = F.sigmoid(model.predict(test_x, test_a)).cpu().detach().numpy()
                    test_auc = skm.roc_auc_score(test_y.cpu().numpy(), test_y_pred)
                    test_acc = skm.accuracy_score(test_y.cpu().numpy(),
                                                  np.where(test_y_pred > 0.5, np.ones_like(test_y_pred),
                                                              np.zeros_like(test_y_pred)))
                    test_logloss = skm.log_loss(test_y.cpu().detach().numpy(), test_y_pred)


                total_test_auc += test_auc
                total_test_acc += test_acc
                total_test_logloss += test_logloss
                total_test_ndcg += test_ndcg
                total_test_recall += test_rec
                total_test_precision += test_pre

            print(
            "Epoch {}\n test_auc:{}, test_acc:{}, test_logloss:{}, test_ndcg:{}, test_recall:{}, test_precision:{}, train_confounder_loss:{}, train_ctr_loss:{}".format(epoch, float(
                total_test_auc / test_dataloader_auc_acc_len), float(total_test_acc / test_dataloader_auc_acc_len), float(
                total_test_logloss / test_dataloader_auc_acc_len), float(
                total_test_ndcg / test_dataloader_len), float(
                total_test_recall / test_dataloader_len), float(
                total_test_precision / test_dataloader_len), float(
                total_confounder_loss / train_dataloader_len), float(
                total_ctr_loss / train_dataloader_len))
                )
            history[epoch][0] = total_test_auc / test_dataloader_len
            history[epoch][1] = total_test_acc / test_dataloader_len
            history[epoch][2] = total_test_logloss / test_dataloader_len
        if epoch % args.iter_save == 0:
            ut.save_model_by_name(model_dir=args.model_dir, model=model, global_step=epoch, history=history)

elif args.debias_mode == 'IPM_Embedding_Sample':
    # training model
    history = np.zeros((args.epoch_max, 3))
    for i in model.parameters():
        i.requires_grad = False
    for i in model.debias_model.prediction_net.parameters():
        i.requires_grad = True

    # define optimizer for max
    max_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.999))
    for i in model.parameters():
        i.requires_grad = True
    for i in model.debias_model.prediction_net.parameters():
        i.requires_grad = False
    min_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.999))

    for epoch in range(args.epoch_max):
        model.train()
        emb_loss_max = 0
        emb_loss_min = 0
        ctr_loss = 0
        total_emb_loss = 0
        total_ctr_loss = 0
        auc = 0
        total_auc = 0
        total_acc = 0
        total_logloss = 0
        # for step in range(args.max_step):
        if args.is_clip_k:
            ipm_sample_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'ipm_data',
                                           'ipm_data_clip{}_{}.npy'.format(args.K2, epoch % 1))
        else:
            ipm_sample_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'ipm_data',
                                           'ipm_data{}_{}.npy'.format(args.K2, epoch % 1))
        train_dataloader = dataload.dataload_ori(train_dataset_dir, args.batch_size,
                                                 data_feature_path=data_feature_path, syn=syn,
                                                 ipm_sample_path=ipm_sample_path)
        for x, a, y, r, x1, a1, y1, r1, x2, a2, y2, r2 in train_dataloader:
            # print(x, a, y, r)
            max_optimizer.zero_grad()

            x, a, y, r, x1, a1, y1, r1, x2, a2, y2, r2 = x.to(device), a.to(device), y.to(device), r.to(device), x1.to(
                device), a1.to(device), y1.to(device), r1.to(device), \
                                                         x2.to(device), a2.to(device), y2.to(device), r2.to(device)
            emb_loss_max = -model.learn(x, a, r, y, epoch, 'max', x1, a1, y1, r1, x2, a2, y2, r2)
            emb_loss_max.backward()
            max_optimizer.step()
            total_emb_loss += emb_loss_max.item()
            m = len(train_dataloader)

            # for step in range(args.min_step):
            # for x, x_bar, a, y in pretrain_dataloader:
            min_optimizer.zero_grad()
            # x, a, y = x.to(device), a.to(device), y.to(device)
            emb_loss_min, ctr_loss = model.learn(x, a, r, y, epoch, 'min', x1, a1, y1, r1, x2, a2, y2, r2)
            L = emb_loss_min + ctr_loss
            L.backward()
            min_optimizer.step()
            total_ctr_loss += L.item()

        if epoch % 1 == 0:
            total_test_auc = 0
            total_test_acc = 0
            total_test_logloss = 0
            for test_x, test_a, test_y, test_r in test_dataloader:
                if args.downstream == 'MLP':
                    test_auc = skm.roc_auc_score(test_y.cpu().numpy(),
                                                 torch.argmax(model.predict(test_x, test_a),
                                                              dim=1).cpu().detach().numpy())
                    test_acc = accuracy_score(test_y.cpu().numpy(),
                                              torch.argmax(model.predict(test_x, test_a), dim=1).cpu().detach().numpy())
                    test_logloss = log_loss(test_y.cpu().numpy(),
                                            F.sigmoid(model.predict(test_x, test_a)[:, 1]).cpu().detach().numpy())
                else:
                    test_y_pred = F.sigmoid(model.predict(test_x, test_a))
                    test_auc = skm.roc_auc_score(test_y.cpu().numpy(), test_y_pred.cpu().detach().numpy())
                    test_acc = skm.accuracy_score(test_y.cpu().numpy(),
                                                  torch.where(test_y_pred > 0.5, torch.ones_like(test_y_pred),
                                                              torch.zeros_like(test_y_pred)).cpu().detach().numpy())
                    test_logloss = skm.log_loss(test_y.cpu().numpy(), test_y_pred.cpu().detach().numpy())
                total_test_auc += test_auc
                total_test_acc += test_acc
                total_test_logloss += test_logloss
            test_dataloader_len = len(test_dataloader)
            train_dataloader_len = len(train_dataloader)
            print(
                "\n test_auc:{}, test_acc:{}, test_logloss:{}".format(float(
                    total_test_auc / test_dataloader_len), float(total_test_acc / test_dataloader_len), float(
                    total_test_logloss / test_dataloader_len)))
            history[epoch][0] = total_test_auc / m
            history[epoch][1] = total_test_acc / m
            history[epoch][2] = total_test_logloss / m
        if epoch % args.iter_save == 0:
            ut.save_model_by_name(model_dir=args.model_dir, model=model, global_step=epoch, history=history)
