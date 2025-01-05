import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils import data
import torch
import os
from PIL import Image

class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        self.image_path = list(map(lambda x: os.path.join(img_root, x), sorted(os.listdir(img_root))))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), sorted(os.listdir(label_root))))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')

        gt = Image.open(self.label_path[item]).convert('L')
        # print(self.image_path[item], self.label_path[item])
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)

class test_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        #image = self.rgb_loader(self.images[self.index])
        image = self.binary_loader(os.path.join(self.image_root,self.img_list[self.index]+ '.png'))
        gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.png'))
        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

