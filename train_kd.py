# coding=utf-8

import argparse
import os
import logging
import torch
import time
import math
import random
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from metrics import *
from plot import *
from Loss import *
from model.AssistNet import *
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from data_loader import get_data

from model.R2UNet import R2U_Net
from model.SegCap import SegCaps
from model.UNet import UNet
from model.UNet16 import UNet_16
from model.NestedUNet import NestedUNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    # parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=61)
    # parse.add_argument('--arch', '-a', metavar='ARCH', default='resnet34_unet',
    #                    help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    # batch_size have to be 1
    parse.add_argument("--batch_size", type=int, default=1)
    # parse.add_argument('--dataset', default='driveEye',  # dsb2018_256
    #                    help='dataset name:liver/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")

    parse.add_argument("--learning_rate", type=float, default=0.001)
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold", type=float, default=None)
    parse.add_argument("--weight_name", default='_1_31', help="name of pth file")

    parse.add_argument("--distill_version", default='fitnet', help="choose from kd, fitnet, at and kc")

    # using experiments to choose a good value for these hyperparameters, have to use real data
    parse.add_argument("--temperature", type=int, default=4)
    parse.add_argument("--alpha", type=float, default=0.05)
    parse.add_argument("--beta", type=float, default=0.005)
    parse.add_argument("--gama", type=float, default=10000)

    # to choose a architecture as teacher model
    # while student model architecture is constant in this pipeline
    parse.add_argument("--teacher_model", default='NestedUNet', help="choose from UNet, NestedUNet and R2U_Net")

    parse.add_argument("-f", help="a dummy argument to fool ipython", default="1")
    args = parse.parse_args()
    return args


def getLog(arg):
    # "arg" should be a input value of this function
    dirname = os.path.join(arg.log_dir, str(arg.batch_size), str(arg.epoch))
    filename = dirname + '/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logging

def fetch_teacher_outputs(teacher_model, dataloader, params):
    teacher_model.eval()
    teacher_outputs = []
    print("now loading teacher's output")
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        # if (torch.cuda.is_available()):
        #     data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        # data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        output_teacher_batch = teacher_model(data_batch)

        # this for would load dic
        for key in output_teacher_batch:
            output_teacher_batch[key] = torch.from_numpy(output_teacher_batch[key].data.cpu().numpy())
            if(torch.cuda.is_available()):
                # output_teacher_batch[key] = Variable(output_teacher_batch[key].cuda(async=True), requires_grad=False)
                output_teacher_batch[key] = output_teacher_batch[key].cuda()
        teacher_outputs.append(output_teacher_batch)
    return teacher_outputs

def evaluate_kd(model, best_iou, val_dataloaders):
    model = model.eval()
    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  # 验证集图片的总数
        # print(num)
        # batch_index,(data,target)
        # x, _,pic,mask
        for data, target in val_dataloaders:
            data = data.to(device)
            y = model(data)['final']
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            hd_total += get_hd(target, img_y)
            miou_total += get_iou(target, img_y)  # 获取当前预测图的miou，并加到总miou中
            dice_total += get_dice(target, img_y)
            if i < num: i += 1  # 处理验证集下一张图
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total / num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou, aver_hd, aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou, aver_hd, aver_dice))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            if not os.path.exists('./saved_dis_model'):
                os.makedirs('./saved_dis_model')
            torch.save(model.state_dict(),
                       r'./saved_dis_model/' + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth')
        return best_iou, aver_iou, aver_dice, aver_hd


def train_kd(model, assist_model, teacher_model, t_s_map_dict, criterion_kd, optimizer, train_dataloader,
             val_dataloader, args):
    best_iou, aver_iou, aver_dice, aver_hd = 0, 0, 0, 0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []

    loading_start = time.time()
    teacher_model.eval()

    teacher_outputs = fetch_teacher_outputs(teacher_model, train_dataloader, args)
    logging.info("- Finished computing teacher outputs after {} secs..".format(math.ceil(time.time() - loading_start)))

    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for i, (x, y) in enumerate(train_dataloader):
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # model
            teacher_out, student_out = assist_model(teacher_outputs[i], model(inputs), t_s_map_dict)
            # loss
            loss = criterion_kd(teacher_out, student_out, labels, t_s_map_dict, args)

            if threshold != None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info(
                "%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        best_iou, aver_iou, aver_dice, aver_hd = evaluate_kd(model, best_iou, val_dataloader)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice', iou_list, dice_list)
    metrics_plot(args, 'hd', hd_list)


if __name__ == '__main__':

    args = getArgs()
    logging = getLog(args)

    # Set the random seed for reproducible experiments, not use for this project
    # random.seed(230)
    # torch.cuda.manual_seed(230) if params.cuda else torch.manual_seed(230)

    ###############
    # Data Loader #
    ###############
    train_loader = get_data(batch_size=args.batch_size, type='train')
    val_loader = get_data(batch_size=args.batch_size, type='val')

    # this temporary data is for construct assist net architecture
    temp_dl = torch.FloatTensor(torch.randn(1, 3, 256, 256))

    print("############################################")
    print("now using " + str(args.distill_version) + " to compress teacher model")
    print("teacher model is " + args.teacher_model)
    print("student model is Unet16")

    ###################
    # Teacher Network #
    ###################
    if (args.teacher_model == 'UNet'):
        teacher_model = UNet(3, 1)
    elif (args.teacher_model == 'NestedUNet'):
        teacher_model = NestedUNet(0, 3, 1)
    elif (args.teacher_model == 'R2U_Net'):
        teacher_model = R2U_Net()
    else:
        raise NotImplementedError(args.teacher_model)

    # this fake output just for create assist net
    teacher_model.eval()
    teacher_out = teacher_model(temp_dl)

    ###################
    # Student Network #
    ###################

    trainable_module_list = nn.ModuleList([])

    student_model = UNet_16(3, 1)

    trainable_module_list.append(student_model)
    student_model.eval()
    student_out = student_model(temp_dl)

    ##################
    # Assist Network #
    ##################

    if (args.distill_version == 'kd'):
        t_s_map_dict = {'teacher': 'logits', 'student': 'logits'}
        assist_model = KD(teacher_out, student_out, t_s_map_dict)
        logging.info('Now using kd Assist Network')
    elif (args.distill_version == 'fitnet'):
        t_s_map_dict = {'teacher': 'feas', 'student': 'feas'}
        assist_model = FitnetConvReg(teacher_out, student_out, t_s_map_dict)
        logging.info('Now using fitnet Assist Network')
    elif (args.distill_version == 'at'):
        t_s_map_dict = {'teacher': 'logits', 'student': 'logits'}
        assist_model = AT(teacher_out, student_out, t_s_map_dict)
        logging.info('Now using at Assist Network')
    elif (args.distill_version == 'kc'):
        t_s_map_dict = {'teacher': 'feas', 'student': 'feas'}
        assist_model = KC(teacher_out, student_out, t_s_map_dict)
        logging.info('Now using KC Assist Network')
    else:
        raise NotImplementedError(args.distill_version)

    assist_model = assist_model.cuda()
    trainable_module_list.append(assist_model)
    # assist_out = assist_model(teacher_out, student_out, t_s_map_dict)

    #############
    #    Loss   #
    #############

    # test assist nets
    if (args.distill_version == 'kd'):
        # loss_fn = Loss.loss_fn_kd
        loss_fn = loss_fn_kd
    elif (args.distill_version == 'fitnet'):
        # loss_fn = Loss.loss_fn_fitnet
        loss_fn = loss_fn_fitnet
    elif (args.distill_version == 'at'):
        # loss_fn = Loss.loss_fn_at
        loss_fn = loss_fn_at
    elif (args.distill_version == 'kc'):
        # loss_fn = Loss.loss_fn_KC
        loss_fn = loss_fn_KC

    ###################
    #    Optimizer    #
    ###################

    optimizer = optim.Adam(trainable_module_list.parameters(), lr=args.learning_rate)

    ########################
    #  load teacher model  #
    ########################

    teacher_model.load_state_dict(torch.load(r'./saved_model/' + str(args.weight_name) + '.pth', map_location='cpu'))

    ######################
    # Run Train and Eval #
    ######################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('**************************')
    print('\nepoch:%s,\nbatch size:%s\n' % \
          (args.epoch, args.batch_size))
    logging.info('\n=======\n\nepoch:%s,\nbatch size:%s\n========' % \
                 (args.epoch, args.batch_size))
    print('**************************')

    student_model.cuda()

    train_kd(student_model, assist_model, teacher_model, t_s_map_dict, loss_fn, optimizer, train_loader, val_loader, args)