# coding=utf-8

import os
import cv2
import json
import torch
import argparse
import os
import logging
import torch.utils.data as data
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skimage import io
from PIL import Image
from thop import profile
from torchvision import transforms
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from model.R2UNet import R2U_Net
from model.SegCap import SegCaps
from model.UNet import UNet
from model.UNet16 import UNet_16
from model.NestedUNet import NestedUNet
from metrics import *
from plot import *


IMG_PATH_BINARY = './drive/MyDrive/dns-panoramic-images-ivisionlab/semantic-segmentation'
IMG_PATH6 = './drive/MyDrive/dns-panoramic-images-ivisionlab/semantic-segmentation/images'
IMG_PATH6_TARGET = './drive/MyDrive/dns-panoramic-images-ivisionlab/semantic-segmentation/masks'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--epoch", type=int, default=61)
    # batch_size have to be 1
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold", type=float, default=None)
    parse.add_argument("--weight_name", default='_1_31', help="name of pth file")
    parse.add_argument("--distill_version", default='kc', help="choose from kd, fitnet, at and kc")
    # to choose a architecture as teacher model
    # while student model architecture is constant in this pipeline
    parse.add_argument("--teacher_model", default='R2U_Net', help="choose from UNet, NestedUNet and R2U_Net")
    parse.add_argument("--test_model", default='UNet16', help="choose from UNet, UNet16, NestedUNet and R2U_Net")
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


class Dataset(data.Dataset):
    def __init__(self, transform_data=None, transform_labels=None):
        self.transform_data, self.transform_labels = transform_data, transform_labels

        names = []
        i = 0
        imgs = []
        labels = []
        for root, dirs, imgs_name in os.walk(IMG_PATH6):
            # only for semantic segmentation training, validation and test

            for img in imgs_name:
                names.append(img)
                img_names = str(root) + '/' + str(img)

                # read images:
                image = cv2.imread(img_names)
                # grayscale, convert to rgb for consistency
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imgs.append(image)

                # print(img_names)

                img = str(img)
                mask = img.replace("jpg", "png")

                img_names = str(IMG_PATH6_TARGET) + '/' + str(mask)
                # print(img_names)
                # read labels:
                label = cv2.imread(img_names)
                label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
                labels.append(label)

        imgs = np.array(imgs)
        labels = np.array(labels)
        self.labels = labels
        self.data = imgs

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        if self.transform_data:
            img = Image.fromarray(img)
            img = self.transform_data(img)
        if self.transform_labels:
            target = Image.fromarray(target)
            target = self.transform_labels(target)
        return img, target

    def __len__(self):
        return len(self.data)


def get_data(batch_size):
    data_loader = data.DataLoader(
        Dataset(
            transform_data=transforms.Compose([
                Resize((256, 256)),
                ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

            ]),
            transform_labels=transforms.Compose([
                Resize((256, 256)),
                ToTensor(),

            ]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    return data_loader


def test_kd(val_dataloaders, model, save_predict=False, distill=False):
    logging.info('final test........')
    if distill == False:
        if save_predict == True:

            dir = os.path.join(r'./saved_dis_predict', str(args.test_model))
            if not os.path.exists(dir):
                os.makedirs(dir)
            else:
                print('dir already exist!')
    else:
        if save_predict == True:

            dir = os.path.join(r'./saved_dis_predict', str(args.distill_version), str(args.teacher_model))
            if not os.path.exists(dir):
                os.makedirs(dir)
            else:
                print('dir already exist!')
    model.eval()

    # plt.ion()
    with torch.no_grad():
        i = 0
        miou_total = 0
        hd_total = 0
        dice_total = 0
        std_d = []
        num = len(val_dataloaders)
        # for pic,_,pic_path,mask_path in val_dataloaders:
        for data, target in val_dataloaders:
            data = data.to(device)
            predict = model(data)['final']
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()
            # img_y = torch.squeeze(y).cpu().numpy()

            iou = get_iou(target, predict)
            miou_total += iou
            hd_total += get_hd(target, predict)
            dice = get_dice(target, predict)
            # std_d is for computing std of dice
            std_d.append(dice)

            dice_total += dice

            # print(target.shape)
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            # show input
            ori_img = data.cpu()
            # print(ori_img.shape)
            # revise shape from (3,512,512) to (512,512,3) for imshow()
            img1_channel1 = ori_img.numpy()[0, :, :, :][0, :, :][:, :, None]  # array(512, 512, 1)
            img1_channel2 = ori_img.numpy()[0, :, :, :][1, :, :][:, :, None]  # array(512, 512, 1)
            img1_channel3 = ori_img.numpy()[0, :, :, :][2, :, :][:, :, None]  # array(512, 512, 1)

            ori_img = np.concatenate((img1_channel1, img1_channel2, img1_channel3), axis=-1)

            plt.imshow(ori_img)

            # print(pic_path[0])
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict, cmap='Greys_r')

            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            # show target
            plt.imshow(target[0][0], cmap='Greys_r')

            # print(mask_path[0])
            if save_predict == True:
                plt.savefig(dir + '/' + str(i) + '.jpg')
            # plt.pause(0.01)
            print('iou={},dice={}'.format(iou, dice))
            if i < num: i += 1
        # plt.show()
        print('Miou=%f,aver_hd=%f,dv=%f,std_dice=%f' % (
        miou_total / num, hd_total / num, dice_total / num, np.std(std_d)))
        logging.info('Miou=%f,aver_hd=%f,dv=%f,std_dice=%f' % (
        miou_total / num, hd_total / num, dice_total / num, np.std(std_d)))
        # print('M_dice=%f' % (dice_total / num))


if __name__ == '__main__':

    args = getArgs()
    logging = getLog(args)

    test_loader = get_data(1)
    temp_dl = torch.FloatTensor(torch.randn(1, 3, 256, 256)).cuda()

    if args.test_model == "UNet16":
        model = UNet_16(3, 1)
    elif args.test_model == "UNet":
        model = UNet(3, 1)
    elif args.test_model == "NestedUNet":
        model = model = NestedUNet(0,in_channel=3,out_channel=1)
    elif args.test_model == "R2U_Net":
        model = R2U_Net()

    model.load_state_dict(
        torch.load(r'./saved_model/' + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth', map_location='cpu'))
    model.cuda()

    flops, params = profile(model, inputs=(temp_dl,), verbose=False)
    logging.info("=" * 100 + "\nnetwork's flops is: {}, parameters count is: {}, params are:".format(flops, params))

    # if we evaluate the result of distillation, distill=True
    test_kd(test_loader, model, save_predict=True, distill=True)
