# 已经将GPU的占用空间压缩到5G bs 2
# 版本中只添加了多尺度一致性损失
# 增加了Synapse数据集训练过程

import argparse
import copy
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from utils import DiceLoss
from torchvision import transforms
from mutils import ramps, losses
from mutils.DataProcess import label_data_index_process
from networks.mosiam import MoCLR, BlockConLoss

def create_no_grad_model(model):
    for param in model.parameters():
        param.detach_()
    return model

def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    # if epoch < args.warmup_epoch:
    #     return 0
    # else:
    return args.consistency * ramps.sigmoid_rampup(epoch-args.warmup_epoch, args.consistency_rampup)

def update_ema_params(model, ema_model, alpha, iter_step):
    alpha = min(1 - 1 / (iter_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))

# def update_ema_params(model, ema_model):
#     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
#         ema_param = param.deepcopy()



def trainer_synapse(args, model, ema_model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from datasets.data_utils import TwoStreamBatchSampler
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    labeled_bs = args.label_batch_size
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs, unlabeled_idxs = label_data_index_process(train_path=args.list_dir, dataset=args.dataset, ratio=args.ratio)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    # 定义ema_model
    ema_model = create_no_grad_model(ema_model)
    mosia_model = MoCLR(model, ema_model, args).cuda()

    mosia_model.train()
    model.train()

    cosin_loss = nn.CosineSimilarity()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    block_con_loss = BlockConLoss()

    optimizer = optim.SGD(mosia_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            unlabeled_image_batch = image_batch[labeled_bs:]
            noise = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.2, 0.2)
            ema_inputs = image_batch + noise
            logits, labels, outputs, ema_outputs = mosia_model(image_batch, ema_inputs)
            ################################
            # 计算encoder部分损失
            v = torch.logsumexp(logits, dim=1, keepdim=True)  # bs 1
            loss_vec = torch.exp(v - v.detach())
            assert loss_vec.shape == (len(logits), 1)
            dummy_logits = torch.cat([torch.zeros(logits.size(0), 1).cuda(), logits], dim=1)
            # # encoder 部分对比损失
            loss_ecs = loss_vec.mean() - 1 + ce_loss(dummy_logits, labels)

            # 一致性损失
            image_batch_r = image_batch.repeat(2, 1, 1, 1)
            stride = image_batch_r.shape[0] // 2  # bs
            ema_image_batch_r = torch.zeros(
                (stride * args.T, image_batch_r.shape[1], image_batch_r.shape[2], image_batch_r.shape[3])).cuda()
            for i in range(args.T // 2):
                ema_image_batch_r[(2 * stride * i):(2 * stride * (i + 1))] = image_batch_r + torch.clamp(
                    torch.randn_like(image_batch_r) * 0.1, -0.2, 0.2)

            with torch.no_grad():
                preds, _, ema_decoder_lst = ema_model(ema_image_batch_r)  # ema_decoder_lst [8xbs, c, w, h] x 3

            ema_decoder_feature = torch.cat(ema_decoder_lst, dim=0)  # [3x8xbs, c, w, h]
            ema_decoder_feature = F.softmax(ema_decoder_feature, dim=1)
            ema_decoder_feature = ema_decoder_feature.reshape(len(ema_decoder_lst), args.T, args.batch_size,
                                                              args.feature_channels, 224, 224)
            ema_decoder_feature = torch.mean(ema_decoder_feature, dim=1)
            ema_decoder_feature_lst = torch.unbind(ema_decoder_feature, dim=0)
            feature_uncertainty_lst = [
                -1.0 * torch.sum(preds_feature_item * torch.log(preds_feature_item + 1e-6), dim=1, keepdim=True)
                for preds_feature_item in ema_decoder_feature_lst]

            # # output级别的不确定性计算
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(args.T, args.batch_size, args.num_classes, 224, 224)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            feature_uncertainty_lst.append(uncertainty)
            # # 权重这里可以做一下处理
            avg_weight = ([1.0 / 6] * 6)
            total_uncertainty = sum([
                item_feature * item_weight for item_feature, item_weight in zip(feature_uncertainty_lst, avg_weight)
            ])
            total_uncertainty = total_uncertainty.cuda()
            consistency_dist = consistency_criterion(outputs, ema_outputs)
            mask = torch.ones(total_uncertainty.shape)
            # # 一致性损失
            consistency_weight = get_current_consistency_weight(epoch_num, args)
            consistency_dist = torch.sum(total_uncertainty * consistency_dist) / (
                        args.num_classes * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist
            # decoder部分有监督对比损失
            features = torch.cat([outputs.unsqueeze(1), ema_outputs.unsqueeze(1)], dim=1)
            decoder_con_loss = block_con_loss(features, label_batch)
            #####################################################################
            loss_ce = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].long())
            loss_dice = dice_loss(outputs[:labeled_bs], label_batch[:labeled_bs], softmax=True)

            loss = (loss_ce + loss_dice + loss_ecs + consistency_loss + decoder_con_loss) / 5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # 更新ema权重参数
            update_ema_params(model, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)


        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def trainer_ACDC(args, model, ema_model, snapshot_path):
    from datasets.dataset_ACDC import ACDC_dataset, RandomGenerator
    from datasets.data_utils import TwoStreamBatchSampler
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info("asdas")
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    labeled_bs = args.label_batch_size
    # max_iterations = args.max_iterations
    db_train = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs, unlabeled_idxs = label_data_index_process(train_path=args.list_dir, dataset=args.dataset, ratio=args.ratio)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ema_model = create_no_grad_model(ema_model)
    mosia_model = MoCLR(model, ema_model, args).cuda()

    mosia_model.train()
    model.train()
    cosin_loss = nn.CosineSimilarity()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    block_con_loss = BlockConLoss()
    optimizer = optim.SGD(mosia_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    warm_iterations = args.warmup_epoch * len(trainloader) #
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            unlabeled_image_batch = image_batch[:labeled_bs] #
            noise = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.2, 0.2)
            ema_inputs = image_batch + noise
            logits, labels, outputs, ema_outputs = mosia_model(image_batch, ema_inputs)
        ################
            # 计算encoder 部分对比损失
            #v = torch.logsumexp(logits, dim=1, keepdim=True)  # bs 1
            #loss_vec = torch.exp(v - v.detach())
            #assert loss_vec.shape == (len(logits), 1)
            #dummy_logits = torch.cat([torch.zeros(logits.size(0), 1).cuda(), logits], dim=1)
            # # encoder 部分对比损失
            #loss_ecs = loss_vec.mean() - 1 + ce_loss(dummy_logits, labels)

            # 一致性损失计算
            image_batch_r = image_batch.repeat(2, 1, 1, 1)
            stride = image_batch_r.shape[0] // 2  # bs
            ema_image_batch_r = torch.zeros((stride * args.T, image_batch_r.shape[1], image_batch_r.shape[2], image_batch_r.shape[3])).cuda()
            for i in range(args.T // 2):
                ema_image_batch_r[(2 * stride * i):(2 * stride * (i + 1))] = image_batch_r + torch.clamp(torch.randn_like(image_batch_r) * 0.1, -0.2, 0.2)

            with torch.no_grad():
                preds, _, ema_decoder_lst = ema_model(ema_image_batch_r) # ema_decoder_lst [8xbs, c, w, h] x 3

            ema_decoder_feature = torch.cat(ema_decoder_lst, dim=0)# [3x8xbs, c, w, h]
            ema_decoder_feature = F.softmax(ema_decoder_feature, dim=1)
            ema_decoder_feature = ema_decoder_feature.reshape(len(ema_decoder_lst), args.T, args.batch_size, args.feature_channels, 224, 224)
            ema_decoder_feature = torch.mean(ema_decoder_feature, dim=1)
            ema_decoder_feature_lst = torch.unbind(ema_decoder_feature, dim=0)
            feature_uncertainty_lst = [-1.0 * torch.sum(preds_feature_item * torch.log(preds_feature_item + 1e-6), dim=1, keepdim=True)
                                       for preds_feature_item in ema_decoder_feature_lst]

            # # output级别的不确定性计算
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(args.T, args.batch_size, args.num_classes, 224, 224)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            feature_uncertainty_lst.append(uncertainty)
            # # 权重这里可以做一下处理
            avg_weight = ([1.0 / 6] * 6)
            total_uncertainty = sum([
                item_feature * item_weight for item_feature, item_weight in zip(feature_uncertainty_lst, avg_weight)
            ])
            total_uncertainty = total_uncertainty.cuda()
            consistency_dist = consistency_criterion(outputs, ema_outputs)
            mask = torch.ones(total_uncertainty.shape)
            # # 一致性损失
            consistency_weight = get_current_consistency_weight(epoch_num, args)
            consistency_dist = torch.sum(total_uncertainty * consistency_dist) / (args.num_classes * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist
            # decoder部分有监督对比损失
            features = torch.cat([outputs.unsqueeze(1), ema_outputs.unsqueeze(1)], dim=1)
            decoder_con_loss = block_con_loss(features, label_batch)
            #####################################################################
            loss_ce = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].long())
            loss_dice = dice_loss(outputs[:labeled_bs], label_batch[:labeled_bs], softmax=True)

            # if epoch_num < args.warmup_epoch:
            #     consistency_loss = 0
            # loss = 0.5 * loss_ce + 0.5 * loss_dice + consistency_loss
            # loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss = (loss_ce + loss_dice + consistency_loss + decoder_con_loss)/5
            # 去掉decoder部分对比的loss
            # loss = (loss_ce + loss_dice + loss_ecs + consistency_loss)/4
            # 去掉decoder 和 encoder对比的loss
            # loss = (loss_ce + loss_dice + consistency_loss)/3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # update_ema_params(model, ema_model, args.ema_decay, iter_num)
            update_ema_params(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # print("this is ", iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f， consist_weight: %f, decoder_con_loss: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), consistency_weight, decoder_con_loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def trainer_la(args, model, ema_model, snapshot_path):
    from datasets.dataset_la import LA_dataset, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, RandomGenerator
    from datasets.data_utils import TwoStreamBatchSampler
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info("asdas")
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    labeled_bs = args.label_batch_size
    # max_iterations = args.max_iterations

    # db_train = LA_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    db_train = LA_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                          transform=transforms.Compose([
                              RandomRotFlip(),
                              RandomCrop((args.img_size, args.img_size)),
                              ToTensor(),
                          ]))

    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs, unlabeled_idxs = label_data_index_process(train_path=args.list_dir, dataset=args.dataset,
                                                            ratio=args.ratio)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ema_model = create_no_grad_model(ema_model)
    mosia_model = MoCLR(model, ema_model, args).cuda()

    mosia_model.train()
    model.train()
    cosin_loss = nn.CosineSimilarity()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    block_con_loss = BlockConLoss()
    optimizer = optim.SGD(mosia_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    warm_iterations = args.warmup_epoch * len(trainloader) #
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            unlabeled_image_batch = image_batch[:labeled_bs] #
            noise = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.2, 0.2)
            ema_inputs = image_batch + noise
            logits, labels, outputs, ema_outputs = mosia_model(image_batch, ema_inputs)
        ################
            # 计算encoder 部分对比损失
            v = torch.logsumexp(logits, dim=1, keepdim=True)  # bs 1
            loss_vec = torch.exp(v - v.detach())
            assert loss_vec.shape == (len(logits), 1)
            dummy_logits = torch.cat([torch.zeros(logits.size(0), 1).cuda(), logits], dim=1)
            # # encoder 部分对比损失
            loss_ecs = loss_vec.mean() - 1 + ce_loss(dummy_logits, labels)

            # 一致性损失计算
            image_batch_r = image_batch.repeat(2, 1, 1, 1)
            stride = image_batch_r.shape[0] // 2  # bs
            ema_image_batch_r = torch.zeros((stride * args.T, image_batch_r.shape[1], image_batch_r.shape[2], image_batch_r.shape[3])).cuda()
            for i in range(args.T // 2):
                ema_image_batch_r[(2 * stride * i):(2 * stride * (i + 1))] = image_batch_r + torch.clamp(torch.randn_like(image_batch_r) * 0.1, -0.2, 0.2)

            with torch.no_grad():
                preds, _, ema_decoder_lst = ema_model(ema_image_batch_r) # ema_decoder_lst [8xbs, c, w, h] x 3

            ema_decoder_feature = torch.cat(ema_decoder_lst, dim=0)# [3x8xbs, c, w, h]
            ema_decoder_feature = F.softmax(ema_decoder_feature, dim=1)
            ema_decoder_feature = ema_decoder_feature.reshape(len(ema_decoder_lst), args.T, args.batch_size, args.feature_channels, args.img_size, args.img_size)
            ema_decoder_feature = torch.mean(ema_decoder_feature, dim=1)
            ema_decoder_feature_lst = torch.unbind(ema_decoder_feature, dim=0)
            feature_uncertainty_lst = [-1.0 * torch.sum(preds_feature_item * torch.log(preds_feature_item + 1e-6), dim=1, keepdim=True)
                                       for preds_feature_item in ema_decoder_feature_lst]

            # # output级别的不确定性计算
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(args.T, args.batch_size, args.num_classes, args.img_size, args.img_size)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            feature_uncertainty_lst.append(uncertainty)
            # # 权重这里可以做一下处理
            avg_weight = ([1.0 / 6] * 6)
            total_uncertainty = sum([
                item_feature * item_weight for item_feature, item_weight in zip(feature_uncertainty_lst, avg_weight)
            ])
            total_uncertainty = total_uncertainty.cuda()
            consistency_dist = consistency_criterion(outputs, ema_outputs)
            mask = torch.ones(total_uncertainty.shape)
            # # 一致性损失
            consistency_weight = get_current_consistency_weight(epoch_num, args)
            consistency_dist = torch.sum(total_uncertainty * consistency_dist) / (args.num_classes * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist
            # decoder部分有监督对比损失
            features = torch.cat([outputs.unsqueeze(1), ema_outputs.unsqueeze(1)], dim=1)
            decoder_con_loss = block_con_loss(features, label_batch)
            #####################################################################
            loss_ce = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].long())
            loss_dice = dice_loss(outputs[:labeled_bs], label_batch[:labeled_bs], softmax=True)

            # if epoch_num < args.warmup_epoch:
            #     consistency_loss = 0
            # loss = 0.5 * loss_ce + 0.5 * loss_dice + consistency_loss
            # loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss = (loss_ce + loss_dice + loss_ecs + consistency_loss + decoder_con_loss)/5
            # 去掉decoder部分对比的loss
            # loss = (loss_ce + loss_dice + loss_ecs + consistency_loss)/4
            # 去掉decoder 和 encoder对比的loss
            # loss = (loss_ce + loss_dice + consistency_loss)/3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # update_ema_params(model, ema_model, args.ema_decay, iter_num)
            update_ema_params(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # print("this is ", iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f， consist_weight: %f, decoder_con_loss: %f, loss_ecs: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), consistency_weight, decoder_con_loss.item(), loss_ecs.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_Lits(args, model, ema_model, snapshot_path):
    from datasets.dataset_lits import LITS_dataset, RandomGenerator
    from datasets.data_utils import TwoStreamBatchSampler
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info("asdas")
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    labeled_bs = args.label_batch_size
    # max_iterations = args.max_iterations
    db_train = LITS_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs, unlabeled_idxs = label_data_index_process(train_path=args.list_dir, dataset=args.dataset, ratio=args.ratio)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ema_model = create_no_grad_model(ema_model)
    mosia_model = MoCLR(model, ema_model, args).cuda()

    mosia_model.train()
    model.train()
    cosin_loss = nn.CosineSimilarity()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    block_con_loss = BlockConLoss()
    optimizer = optim.SGD(mosia_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    warm_iterations = args.warmup_epoch * len(trainloader) #
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            unlabeled_image_batch = image_batch[:labeled_bs] #
            noise = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.2, 0.2)
            ema_inputs = image_batch + noise
            logits, labels, outputs, ema_outputs = mosia_model(image_batch, ema_inputs)
        ################
            # 计算encoder 部分对比损失
            v = torch.logsumexp(logits, dim=1, keepdim=True)  # bs 1
            loss_vec = torch.exp(v - v.detach())
            assert loss_vec.shape == (len(logits), 1)
            dummy_logits = torch.cat([torch.zeros(logits.size(0), 1).cuda(), logits], dim=1)
            # # encoder 部分对比损失
            loss_ecs = loss_vec.mean() - 1 + ce_loss(dummy_logits, labels)

            # 一致性损失计算
            image_batch_r = image_batch.repeat(2, 1, 1, 1)
            stride = image_batch_r.shape[0] // 2  # bs
            ema_image_batch_r = torch.zeros((stride * args.T, image_batch_r.shape[1], image_batch_r.shape[2], image_batch_r.shape[3])).cuda()
            for i in range(args.T // 2):
                ema_image_batch_r[(2 * stride * i):(2 * stride * (i + 1))] = image_batch_r + torch.clamp(torch.randn_like(image_batch_r) * 0.1, -0.2, 0.2)

            with torch.no_grad():
                preds, _, ema_decoder_lst = ema_model(ema_image_batch_r) # ema_decoder_lst [8xbs, c, w, h] x 3

            ema_decoder_feature = torch.cat(ema_decoder_lst, dim=0)# [3x8xbs, c, w, h]
            ema_decoder_feature = F.softmax(ema_decoder_feature, dim=1)
            ema_decoder_feature = ema_decoder_feature.reshape(len(ema_decoder_lst), args.T, args.batch_size, args.feature_channels, 224, 224)
            ema_decoder_feature = torch.mean(ema_decoder_feature, dim=1)
            ema_decoder_feature_lst = torch.unbind(ema_decoder_feature, dim=0)
            feature_uncertainty_lst = [-1.0 * torch.sum(preds_feature_item * torch.log(preds_feature_item + 1e-6), dim=1, keepdim=True)
                                       for preds_feature_item in ema_decoder_feature_lst]

            # # output级别的不确定性计算
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(args.T, args.batch_size, args.num_classes, 224, 224)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            feature_uncertainty_lst.append(uncertainty)
            # # 权重这里可以做一下处理
            avg_weight = ([1.0 / 6] * 6)
            total_uncertainty = sum([
                item_feature * item_weight for item_feature, item_weight in zip(feature_uncertainty_lst, avg_weight)
            ])
            total_uncertainty = total_uncertainty.cuda()
            consistency_dist = consistency_criterion(outputs, ema_outputs)
            mask = torch.ones(total_uncertainty.shape)
            # # 一致性损失
            consistency_weight = get_current_consistency_weight(epoch_num, args)
            consistency_dist = torch.sum(total_uncertainty * consistency_dist) / (args.num_classes * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist
            # decoder部分有监督对比损失
            features = torch.cat([outputs.unsqueeze(1), ema_outputs.unsqueeze(1)], dim=1)
            decoder_con_loss = block_con_loss(features, label_batch)
            #####################################################################
            loss_ce = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].long())
            loss_dice = dice_loss(outputs[:labeled_bs], label_batch[:labeled_bs], softmax=True)

            # if epoch_num < args.warmup_epoch:
            #     consistency_loss = 0
            # loss = 0.5 * loss_ce + 0.5 * loss_dice + consistency_loss
            # loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss = (loss_ce + loss_ecs + loss_dice + consistency_loss + decoder_con_loss)/5
            # 去掉decoder部分对比的loss
            # loss = (loss_ce + loss_dice + loss_ecs + consistency_loss)/4
            # 去掉decoder 和 encoder对比的loss
            # loss = (loss_ce + loss_dice + consistency_loss)/3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # update_ema_params(model, ema_model, args.ema_decay, iter_num)
            update_ema_params(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # print("this is ", iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f， consist_weight: %f, decoder_con_loss: %f， loss_ecs： %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), consistency_weight, decoder_con_loss.item(), loss_ecs.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


# def trainer_ACDC(args, model, snapshot_path):
#     from datasets.dataset_ACDC import ACDC_dataset, RandomGenerator
#     from datasets.data_utils import TwoStreamBatchSampler
#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu
#     labeled_bs = args.label_batch_size
#     # max_iterations = args.max_iterations
#     db_train = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
#                                transform=transforms.Compose(
#                                    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
#     print("The length of train set is: {}".format(len(db_train)))
#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)
#
#     labeled_idxs, unlabeled_idxs = label_data_index_process(train_path=args.list_dir,dataset=args.dataset)
#     batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
#
#     # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
#     #                          worker_init_fn=worker_init_fn)
#     trainloader = DataLoader(db_train, batch_sampler=batch_sampler,  num_workers=8, pin_memory=True,
#                              worker_init_fn=worker_init_fn)
#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)
#
#     model.train()
#     ce_loss = CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#
#     writer = SummaryWriter(snapshot_path + '/log')
#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
#     logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
#     best_performance = 0.0
#     # model.train()
#     iterator = tqdm(range(max_epoch), ncols=70)
#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#             outputs = model(image_batch)
#             loss_ce = ce_loss(outputs, label_batch[:].long())
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#
#             loss = 0.5 * loss_ce + 0.5 * loss_dice
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_
#
#             iter_num = iter_num + 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)
#
#             logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
#
#             # if iter_num % 20 == 0:
#             #     image = image_batch[1, 0:1, :, :]
#             #     image = (image - image.min()) / (image.max() - image.min())
#             #     writer.add_image('train/Image', image, iter_num)
#             #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#             #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#             #     labs = label_batch[1, ...].unsqueeze(0) * 50
#             #     writer.add_image('train/GroundTruth', labs, iter_num)
#
#
#         save_interval = 50  # int(max_epoch/6)
#         if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
#             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#
#
#
#         if epoch_num >= max_epoch - 1:
#             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#             iterator.close()
#             break
#
#     writer.close()
#     return "Training Finished!"
