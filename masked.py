import torch
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

# 加载图片
# img = Image.open('0.jpg')
#
# # 图像变换，包括将图像转换为张量，将像素值归一化等
# transform = transforms.Compose([
#     transforms.Resize((384, 384)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# img = transform(img)
# img = img.view(1, img.shape[0], img.shape[1], img.shape[2])

def maske_aug(img):
# 将图像分割成16块
    h, w = img.shape[2], img.shape[3]
    crop_size = min(h, w) // 8  # 每个块的大小
    num_blocks = 64
    img_blocks = []
    for i in range(num_blocks):
        row = (i // 8) * crop_size
        col = (i % 8) * crop_size
        block = img[:, :, row:row+crop_size, col:col+crop_size]
        img_blocks.append(block)

    for i in range(8):
        rand_block_idx = np.random.randint(0, len(img_blocks))
        rand_block = img_blocks[rand_block_idx]
        if i % 2 ==0:
            noisy_block = rand_block * 0
        else:
            random_number = random.randint(5, 100)
            noise = torch.randn_like(rand_block) * random_number  # 生成均值为0、标准差为0.1的噪声
            noisy_block = rand_block + noise
        img_blocks[rand_block_idx] = noisy_block
    # img_blocks = img_blocks[:rand_block_idx] + (noisy_block,) + img_blocks[rand_block_idx+1:]

    # 重新组合图像块
    noisy_img1 = torch.cat((img_blocks[0:8]), dim=3)
    noisy_img2 = torch.cat((img_blocks[8:16]), dim=3)
    noisy_img3 = torch.cat((img_blocks[16:24]), dim=3)
    noisy_img4 = torch.cat((img_blocks[24:32]), dim=3)
    noisy_img5 = torch.cat((img_blocks[32:40]), dim=3)
    noisy_img6 = torch.cat((img_blocks[40:48]), dim=3)
    noisy_img7 = torch.cat((img_blocks[48:56]), dim=3)
    noisy_img8 = torch.cat((img_blocks[56:64]), dim=3)
    noisy_img = torch.cat((noisy_img1, noisy_img2, noisy_img3, noisy_img4, noisy_img5, noisy_img6, noisy_img7, noisy_img8), dim=2)

    return noisy_img

# def maske_aug(img):
# # 将图像分割成16块
#     h, w = img.shape[2], img.shape[3]
#     crop_size = min(h, w) // 4  # 每个块的大小
#     num_blocks = 16
#     img_blocks = []
#     for i in range(num_blocks):
#         row = (i // 4) * crop_size
#         col = (i % 4) * crop_size
#         block = img[:, :, row:row+crop_size, col:col+crop_size]
#         img_blocks.append(block)
#
#     rand_block_idx = np.random.randint(0, len(img_blocks))
#     rand_block = img_blocks[rand_block_idx]
#     random_number = random.randint(5, 100)
#     noise = torch.randn_like(rand_block) * random_number  # 生成均值为0、标准差为0.1的噪声
#     noisy_block = rand_block + noise
#     img_blocks[rand_block_idx] = noisy_block
#     # img_blocks = img_blocks[:rand_block_idx] + (noisy_block,) + img_blocks[rand_block_idx+1:]
#
#     # 重新组合图像块
#     noisy_img1 = torch.cat((img_blocks[0:4]), dim=3)
#     noisy_img2 = torch.cat((img_blocks[4:8]), dim=3)
#     noisy_img3 = torch.cat((img_blocks[8:12]), dim=3)
#     noisy_img4 = torch.cat((img_blocks[12:]), dim=3)
#     noisy_img = torch.cat((noisy_img1, noisy_img2, noisy_img3, noisy_img4), dim=2)
#
#     return noisy_img

# 展示图像
# plt.imshow(noisy_img[0].permute(1, 2, 0))
# plt.show()
# 在一个随机的块上添加噪声
# rand_block_idx = random.randint(0, num_blocks - 1)
# rand_block = blocks[rand_block_idx]
# noise = torch.randn_like(rand_block) * 0.1  # 噪声的标准差为0.1
# noisy_block = rand_block + noise
# blocks[rand_block_idx] = noisy_block

# 将块重新组合成一个图像
# noisy_img = torch.cat(blocks, dim=2)
#
# # 显示原始图像和添加噪声后的图像
# plt.subplot(121)
# plt.imshow(img.permute(1, 2, 0))
# plt.title('Original Image')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(noisy_img.permute(1, 2, 0))
# plt.title('Noisy Image')
# plt.axis('off')
# plt.show()




#
#
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 加载图像
# img_path = "0.jpg"
# img = Image.open(img_path)
#
# # 对图像进行变换
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])
# img = transform(img)
#
# # 将图像分成 16 块
# img_blocks = torch.chunk(img, 16, dim=(2,3))
#
# # 随机选择一个块并添加噪声
# rand_block_idx = np.random.randint(0, len(img_blocks))
# rand_block = img_blocks[rand_block_idx]
# noise = torch.randn_like(rand_block) * 1  # 生成均值为0、标准差为0.1的噪声
# noisy_block = rand_block + noise
# img_blocks = img_blocks[:rand_block_idx] + (noisy_block,) + img_blocks[rand_block_idx+1:]
#
# # 重新组合图像块
# noisy_img = torch.cat(img_blocks, dim=2)
#
# # 展示图像
# plt.imshow(noisy_img.permute(1, 2, 0))
# plt.show()
