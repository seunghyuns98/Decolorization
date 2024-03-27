from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random
from itertools import chain


class Step1(Dataset):
    def __init__(self, root_dir):
        self.classes = ['Expert_A', 'Expert_C', 'Input', 'Expert_B']
        self.objects = ['nonhuman', 'human', 'building', 'nature']

        building = [_ for _ in os.listdir(os.path.join(root_dir, 'step1', 'Input', 'building')) if
                    _.endswith('.jpg') or _.endswith('.JPG')]
        nonhuman = [_ for _ in os.listdir(os.path.join(root_dir, 'step1', 'Input', 'nonhuman')) if
                    _.endswith('.jpg') or _.endswith('.JPG')]
        human = [_ for _ in os.listdir(os.path.join(root_dir, 'step1', 'Input', 'human')) if
                 _.endswith('.jpg') or _.endswith('.JPG')]
        nature = [_ for _ in os.listdir(os.path.join(root_dir, 'step1', 'Input', 'nature')) if
                  _.endswith('.jpg') or _.endswith('.JPG')]
        self.image_names = [nonhuman, human, building, nature]
        self.root_dir = root_dir + '/step1'

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)])

        self.n_bu = len(building)
        self.n_hu = len(human)
        self.n_nh = len(nonhuman)
        self.n_na = len(nature)

        self.num_classes = len(self.classes)
        self.num_images = self.n_bu + self.n_hu + self.n_nh + self.n_na

    def __len__(self):
        return self.num_images * self.num_classes

    def __getitem__(self, idx):
        style_l = idx // self.num_images
        images_l = idx % self.num_images

        if (idx % self.num_images) < self.n_nh:
            objects_l = 0
        elif idx % self.num_images < (self.n_nh + self.n_hu):
            objects_l = 1
            images_l = images_l - self.n_nh
        elif idx % self.num_images < (self.n_nh + self.n_hu + self.n_bu):
            objects_l = 2
            images_l = images_l - self.n_nh - self.n_hu
        else:
            objects_l = 3
            images_l = images_l - self.n_nh - self.n_hu - self.n_bu

        path_n = self.classes[style_l]
        objects_n = self.objects[objects_l]
        image_n = self.image_names[objects_l][images_l]

        image_path = os.path.join(self.root_dir, path_n, objects_n, image_n)
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        label = style_l * len(self.objects) + objects_l

        return image, style_l, objects_l, label


