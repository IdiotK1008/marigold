import h5py
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# 加载数据
file_path = 'data/nyu/nyuv2.mat'
with h5py.File(file_path, 'r') as f:
    rgb_images = f['images'][:]  # RGB 图像
    depth_images = f['rawDepths'][:]  # 深度图像
    filled_images = f['depths'][:]  # 填充深度图像
    rgb_references = f['rawRgbFilenames']  # RGB 文件名引用
    rgb_filenames = [
        ''.join(chr(c[0]) for c in f[ref][()])
        for ref in rgb_references[0]
    ]

output_dir = 'data/nyu'
for i, filename in tqdm(enumerate(rgb_filenames), total=len(rgb_filenames), desc="Saving images"):
    image = rgb_images[i,:,:,:]
    image = np.transpose(image, (2,1,0))
    depth = depth_images[i]
    depth = np.transpose(depth)
    filled = filled_images[i]
    filled = np.transpose(filled)
    name = filename.split('/')[0]

    # 保存图像
    image_path = os.path.join(output_dir, name)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    j = i+1
    rgb_path = os.path.join(image_path, f'rgb_{j:04d}.png')
    depth_path = os.path.join(image_path, f'depth_{j:04d}.png')
    filled_path = os.path.join(image_path, f'filled_{j:04d}.png')

    # 使用 PIL 保存图像
    rgb_image = Image.fromarray(image)
    rgb_image.save(rgb_path)

    # 将深度图像转换为 uint8 格式
    depth_image = Image.fromarray((depth * 1000).astype(np.uint16))
    depth_image.save(depth_path)

    # 将填充深度图像转换为 uint8 格式
    filled_image = Image.fromarray((filled * 1000).astype(np.uint16))
    filled_image.save(filled_path)

    # break  # 仅保存第一张图像，测试用
