from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random


class Step2(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.classes = ['Expert_A', 'Expert_C', 'Input', 'Expert_B']
        self.objects = ['nonhuman', 'human', 'building', 'nature']
        self.mode = mode
        self.resize = transforms.Resize((512, 512))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)])
        self.split = split

        if split == 'train':

            building = [_ for _ in os.listdir('./dataset/BW_Adobe5k/train/Input/building') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            nonhuman = [_ for _ in os.listdir('./dataset/BW_Adobe5k/train/Input/nonhuman') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            human = [_ for _ in os.listdir('./dataset/BW_Adobe5k/train/Input/human') if
                     _.endswith('.jpg') or _.endswith('.JPG')]
            nature = [_ for _ in os.listdir('./datasetBW_Adobe5k/train/Input/nature') if
                      _.endswith('.jpg') or _.endswith('.JPG')]
            self.image_names = [nonhuman, human, building, nature]

        else:

            building = [_ for _ in os.listdir('./dataset/BW_Adobe5k/test/Input/building') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            nonhuman = [_ for _ in os.listdir('./dataset/BW_Adobe5k/test/Input/nonhuman') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            human = [_ for _ in os.listdir('./dataset/BW_Adobe5k/test/Input/human') if
                     _.endswith('.jpg') or _.endswith('.JPG')]
            nature = [_ for _ in os.listdir('./dataset/BW_Adobe5k/test/Input/nature') if
                      _.endswith('.jpg') or _.endswith('.JPG')]

        self.image_names = [nonhuman, human, building, nature]

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

        gt_path = os.path.join(self.root_dir, path_n, objects_n, image_n)
        image_path = os.path.join(self.root_dir, 'Input', objects_n, image_n)

        rgb_img = Image.open(image_path).convert('RGB')
        rgb_img = np.array(self.resize(rgb_img))
        rgb_img = self.transform(rgb_img)

        gray_img = Image.open(image_path).convert('L')
        gray_img = np.array(self.resize(gray_img))
        gray_img = self.transform(gray_img)

        gt_img = Image.open(gt_path).convert('L')
        gt_img = np.array(self.resize(gt_img))
        gt_img = self.transform(gt_img)

        label = style_l * len(self.objects) + objects_l

        return rgb_img, gt_img, gray_img, image_n, label

