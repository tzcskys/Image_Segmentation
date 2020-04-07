# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import SimpleITK as sitk
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


class Fudan_2D_Dataset(Dataset):

    def __init__(self, csv_file, phase, flip_rate=0):
        self.data = pd.read_csv(csv_file)
        self.flip_rate = flip_rate
        if phase == 'val':
            self.flip_rate = 0.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data['ct_image'][idx]
        img = sitk.ReadImage(img_name)
        img = sitk.GetArrayFromImage(img)
        img = img / 255.

        label_name = self.data['ct_msk'][idx]
        label = sitk.ReadImage(label_name)
        label = sitk.GetArrayFromImage(label)
        label = label / 255.

        if random.random() < self.flip_rate:
            img = np.fliplr(img)
            label = np.fliplr(label)

        # zero centering
        # img = img[:, :, ::-1]
        # switch to BGR
        # img = np.transpose(img, (2, 0, 1)) / 255.
        # img[0] -= self.means[0]
        # img[1] -= self.means[1]
        # img[2] -= self.means[2]

        # the input of unet is set to 3 channels
        img = np.expand_dims(img, 0).repeat(3, axis=0)
        img_CT = torch.from_numpy(img).float()

        # The target should be a LongTensor using nn.CrossEntropyLoss (or nn.NLLLoss)
        label_CT = torch.from_numpy(label).float()

        # sample1 = {'X': img_CT, 'l': label_CT}
        # return sample1
        return img_CT, label_CT

# def show_batch(batch):
#     img_batch = batch['X']
#     # img_batch[:,0,...].add_(means[0])
#     # img_batch[:,1,...].add_(means[1])
#     # img_batch[:,2,...].add_(means[2])
#
#     grid = utils.make_grid(img_batch)
#     # the array axis is [C,H,W], if you want to plot, you should transpose to [H,W,C]
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))
#
#     plt.title('Batch from dataloader')
#
#
# if __name__ == "__main__":
#
#     root_dir = "./"
#     train_file = os.path.join(root_dir, "train.csv")
#
#     # means = np.array([18.8634, 18.8634, 18.8634]) / 255.
#
#     train_data = Fudan_2D_Dataset(csv_file=train_file, phase='train')
#
#     # show a batch
#     batch_size = 4
#     for i in range(batch_size):
#         sample = train_data[i]
#         print(str(i) + str(sample['X'].size()))
#
#     dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
#
#     for i, batch in enumerate(dataloader):
#
#         print(i, batch['X'].size())
#         # observe 4th batch
#         if i == 3:
#             plt.figure()
#             show_batch(batch)
#             plt.axis('off')
#             plt.ioff()
#             plt.show()
