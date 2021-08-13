# coding=utf-8

# batch_size should be set as 1
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from scipy.spatial.distance import directed_hausdorff


def get_iou(mask, predict):
    # image_mask = cv2.imread(mask_name,0)
    # if np.all(image_mask == None):
    #     image_mask = imageio.mimread(mask_name)
    #     image_mask = np.array(image_mask)[0]
    #     image_mask = cv2.resize(image_mask,(576,576))

    # print(image.shape)

    image_mask = torch.squeeze(mask).cpu().numpy()
    height = predict.shape[0]
    weight = predict.shape[1]
    # print(height*weight)
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1

    predict = predict.astype(np.int16)

    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    iou_tem = inter / union

    # print('%s:iou=%f' % ("iou",iou_tem))

    return iou_tem


def get_dice(mask, predict):
    # image_mask = cv2.imread(mask_name, 0)
    # if np.all(image_mask == None):
    #     image_mask = imageio.mimread(mask_name)
    #     image_mask = np.array(image_mask)[0]
    #     image_mask = cv2.resize(image_mask,(576,576))

    image_mask = torch.squeeze(mask).cpu().numpy()
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1

    predict = predict.astype(np.int16)
    intersection = (predict * image_mask).sum()
    dice = (2. * intersection) / (predict.sum() + image_mask.sum())
    return dice


def get_hd(mask, predict):
    # image_mask = cv2.imread(mask_name, 0)
    # # print(mask_name)
    # # print(image_mask)
    # if np.all(image_mask == None):
    #     image_mask = imageio.mimread(mask_name)
    #     image_mask = np.array(image_mask)[0]
    #     image_mask = cv2.resize(image_mask,(576,576))
    # print(mask.shape)
    # print(mask)
    image_mask = torch.squeeze(mask).cpu().numpy()
    # print(mask.shape)
    # print("####################")
    # print(predict.shape)
    # print(predict)

    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0

    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1

    hd1 = directed_hausdorff(image_mask, predict)[0]
    hd2 = directed_hausdorff(predict, image_mask)[0]
    res = None
    if hd1 > hd2 or hd1 == hd2:
        res = hd1
        return res
    else:
        res = hd2
        return res

