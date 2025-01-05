import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance

def cv_random_flip(img, label, body, edge, t):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        body = body.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        t = t.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     t = t.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, body, edge, t


def randomCrop(image, label, body, edge, t):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), body.crop(random_region), edge.crop(random_region), t.crop(random_region)


def randomRotation(image, label, body, edge, t):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        body = body.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
        t = t.rotate(random_angle, mode)
    return image, label, body, edge, t


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized t maps for training and test. If you use the normalized t maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, root_path, trainsize):
        self.trainsize = trainsize
        self.images = []
        self.gts = []
        self.bodys = []
        self.edges = []
        self.ts = []
        lines = os.listdir(os.path.join(image_root, 'GT'))
        for line in lines:
            self.images.append(os.path.join(root_path, 'RGB/', line[:-4] + '.jpg'))
            self.gts.append(os.path.join(image_root, 'GT/', line))
            self.bodys.append(os.path.join(root_path, 'Body/', line))
            self.edges.append(os.path.join(root_path, 'Edge/', line))
            self.ts.append(os.path.join(root_path, 'D/', line[:-4] + '.png'))
        # self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        #
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        #
        # self.ts = [t_root + f for f in os.listdir(t_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.bodys = sorted(self.bodys)
        self.edges = sorted(self.edges)
        self.ts = sorted(self.ts)
        # self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.ts_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor(), transforms.Normalize([0.241, 0.236, 0.244], [0.208, 0.269, 0.241])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        body = self.binary_loader(self.bodys[index])
        edge = self.binary_loader(self.edges[index])
        t = self.rgb_loader(self.ts[index])  # RGBT

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        image, gt, body, edge, t = cv_random_flip(image, gt, body, edge, t)
        image, gt, body, edge, t = randomCrop(image, gt, body, edge, t)
        image, gt, body, edge, t = randomRotation(image, gt, body, edge, t)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        body = self.gt_transform(body)
        edge = self.gt_transform(edge)
        t = self.ts_transform(t)
        return image, t, gt, body, edge

    # def filter_files(self):
    #     assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
    #     images = []
    #     gts = []
    #     ts = []
    # 
    #     for img_path, gt_path, t_path in zip(self.images, self.gts, self.ts):
    #         img = Image.open(img_path)
    #         gt = Image.open(gt_path)
    #         t = Image.open(t_path)
    #         if img.size == gt.size and gt.size == t.size:
    #             images.append(img_path)
    #             gts.append(gt_path)
    #             ts.append(t_path)
    #     self.images = images
    #     self.gts = gts
    #     self.ts = ts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    # def resize(self, img, gt, t):
    #     assert img.size == gt.size and gt.size == t.size
    #     w, h = img.size
    #     if h < self.trainsize or w < self.trainsize:
    #         h = max(h, self.trainsize)
    #         w = max(w, self.trainsize)
    #         return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
    #                t.resize((w, h),Image.NEAREST)
    #     else:
    #         return img, gt, t

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, root_path, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
    dataset = SalObjDataset(image_root, root_path, trainsize)
    # print(image_root)
    # print(gt_root)
    # print(t_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, gt_root, root_path, testsize):
        self.testsize = testsize
        self.images = []
        self.gts = []
        self.ts = []
        lines = os.listdir(os.path.join(gt_root, 'GT/'))
        for line in lines:
            self.images.append(os.path.join(root_path, 'RGB/', line[:-4] + '.jpg'))
            self.gts.append(os.path.join(gt_root, 'GT/', line))
            self.ts.append(os.path.join(root_path, 'D/', line[:-4] + '.png'))
        # self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.ts = [t_root + f for f in os.listdir(t_root) if f.endswith('.png') or f.endswith('.jpg')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.ts = sorted(self.ts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.ts_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor(), transforms.Normalize([0.493, 0.493, 0.493], [0.231, 0.231, 0.231])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        t = self.rgb_loader(self.ts[self.index]) # RGBT
        t = self.transform(t).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, t, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

