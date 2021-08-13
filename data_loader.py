# coding=utf-8

import os
import cv2
import json
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np

from utils.conversions import annotations2mask, mask2instances
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from skimage import io


# all directions should be revised according to the specific local
IMG_PATH = './drive/MyDrive/dns-panoramic-images-ivisionlab/instance-segmentation/folds'

IMG_PATH1 = './drive/MyDrive/dns-panoramic-images-ivisionlab/instance-segmentation/folds/fold1/images'
IMG_PATH2 = './drive/MyDrive/dns-panoramic-images-ivisionlab/instance-segmentation/folds/fold2/images'
IMG_PATH3 = './drive/MyDrive/dns-panoramic-images-ivisionlab/instance-segmentation/folds/fold3/images'
IMG_PATH4 = './drive/MyDrive/dns-panoramic-images-ivisionlab/instance-segmentation/folds/fold4/images'
IMG_TARGET = './drive/MyDrive/binary_masks/'

IMG_PATH5 = './drive/MyDrive/dns-panoramic-images-ivisionlab/instance-segmentation/folds/fold5/images'
JSON_FILEPATH = './drive/MyDrive/dns-panoramic-images-ivisionlab/instance-segmentation/complete-json/dns-panoramic-images.json'

IMG_PATH_BINARY = './drive/MyDrive/dns-panoramic-images-ivisionlab/semantic-segmentation'
IMG_PATH6 = './drive/MyDrive/dns-panoramic-images-ivisionlab/semantic-segmentation/images'
IMG_PATH6_TARGET = './drive/MyDrive/dns-panoramic-images-ivisionlab/semantic-segmentation/masks'


json_file = open(JSON_FILEPATH, 'r')
json_data = json.load(json_file)

images = json_data['images']
annotations = json_data['annotations']


# only for training and validation dataset, data for test should be in test.py
class Dataset(data.Dataset):
    def __init__(self, file_dir, transform_data=None, transform_labels=None, type='train'):
        self.transform_data,self.transform_labels=transform_data,transform_labels
        imgs = []
        labels = []
        for root, dirs, imgs_name in os.walk(file_dir):
          # only for semantic segmentation training, validation and test
          if type == 'train':
            if ((root == IMG_PATH1)|(root == IMG_PATH2)|(root == IMG_PATH3)|(root == IMG_PATH4)):
              for img in imgs_name:
                img_names = str(root)+'/'+ str(img)

                # read images:
                image = cv2.imread(img_names)
                # grayscale, convert to rgb for consistency
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imgs.append(image)

                # read labels:
                sample_image_json = list(filter(lambda x: x['file_name'] == img, images))[0]
                sample_image_id = sample_image_json['id']
                # get the image annotations
                sample_annotations = [ann for ann in annotations if ann['image_id'] == sample_image_id]
                # get non-binary and binary mask
                height, width = sample_image_json['height'], sample_image_json['width']
                # teeth are white and background is black
                mask = annotations2mask(sample_annotations, height, width, binary=True)
                labels.append(mask)

          elif type == 'val':
            if (root == IMG_PATH5):
              for img in imgs_name:
                img_names = str(root)+'/'+ str(img)

                # read images:
                image = cv2.imread(img_names)
                # grayscale, convert to rgb for consistency
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imgs.append(image)

                # read labels:
                sample_image_json = list(filter(lambda x: x['file_name'] == img, images))[0]
                sample_image_id = sample_image_json['id']
                # get the image annotations
                sample_annotations = [ann for ann in annotations if ann['image_id'] == sample_image_id]
                # get non-binary and binary mask
                height, width = sample_image_json['height'], sample_image_json['width']
                # teeth are white and background is black
                mask = annotations2mask(sample_annotations, height, width, binary=True)
                labels.append(mask)


        imgs = np.array(imgs)
        labels = np.array(labels)
        self.labels=labels
        self.data=imgs

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        if self.transform_data:
            img = Image.fromarray(img)
            img=self.transform_data(img)
        if self.transform_labels:
            target = Image.fromarray(target)
            target=self.transform_labels(target)
        return img, target

    def __len__(self):
        return len(self.data)


def get_data(batch_size, type, data_root=IMG_PATH):
    data_loader = data.DataLoader(
        Dataset(
            file_dir=data_root,
            transform_data=transforms.Compose([
                Resize((256, 256)),
                ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

            ]),
            transform_labels=transforms.Compose([
                Resize((256, 256)),
                ToTensor(),

            ]),
            type=type
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    return data_loader

