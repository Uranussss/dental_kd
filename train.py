# coding=uft-8

import json
import argparse
import os
import logging
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from metrics import *
from plot import *
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
    parse.add_argument("--epoch", type=int, default=31)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='unet',
                       help='unet/r2u_net/unet++/SegCaps/unet_16')
    # batch_size have to be 1
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--learning_rate",type=float,default=1e-3)
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    parse.add_argument("-f", help="a dummy argument to fool ipython", default="1")
    args = parse.parse_args()
    return args

def getLog(arg):
    # "arg" should be a input value of this function
    dirname = os.path.join(arg.log_dir,str(arg.batch_size),str(arg.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def val(model,best_iou,val_dataloaders):
    model= model.eval()
    with torch.no_grad():
        i=0
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)
        #print(num)

        for data,target in val_dataloaders:
            data = data.to(device)
            ##
            # for others
            y = model(data)['final']
            # for segcap training
            # y = model(data)

            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()

            hd_total += get_hd(target, img_y)
            miou_total += get_iou(target,img_y)
            dice_total += get_dice(target,img_y)
            if i < num:i+=1
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total/num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            if not os.path.exists('./saved_model'):
                os.makedirs('./saved_model')
            torch.save(model.state_dict(), r'./saved_model/'+'_'+str(args.batch_size)+'_'+str(args.epoch)+'.pth')
        return best_iou,aver_iou,aver_dice,aver_hd


def train(model, criterion, optimizer, train_dataloader, val_dataloader, args):
    best_iou, aver_iou, aver_dice, aver_hd = 0, 0, 0, 0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                ##
                outputs = model(inputs)['final']
                # outputs = model(inputs)

                loss = 0

                # print(output)
                # print(output.shape)
                # print(labels)
                # print(labels.shape)

                for output in outputs:
                    loss += criterion(output, labels)
                loss /= len(outputs)
            else:
                output = model(inputs)['final']
                # output = model(inputs)
                # print(output)
                # print(output.shape)
                # print(labels)
                # print(labels.shape)

                loss = criterion(output, labels)
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

        best_iou, aver_iou, aver_dice, aver_hd = val(model, best_iou, val_dataloader)
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

    train_loader = get_data(batch_size=args.batch_size, type='train')
    val_loader = get_data(batch_size=args.batch_size, type='val')

    print('**************************')
    print('\nepoch:%s,\nbatch size:%s\n' % \
          (args.epoch, args.batch_size))
    logging.info('\n=======\n\nepoch:%s,\nbatch size:%s\n========' % \
                 (args.epoch, args.batch_size))
    print('**************************')

    # 'unet/r2u_net/unet++/SegCaps/unet_16'
    if (args.arch == 'unet'):
        model = UNet(3,1)
    elif (args.arch == 'unet_16'):
        model = UNet_16(3,1)
    elif (args.arch == 'unet++'):
        model = NestedUNet(0,in_channel=3,out_channel=1)
    elif (args.arch == 'r2u_net'):
        model = R2U_Net()
    elif (args.arch == 'SegCaps'):
        model = SegCaps()

    model.cuda()

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train(model, criterion, optimizer, train_loader, val_loader, args)