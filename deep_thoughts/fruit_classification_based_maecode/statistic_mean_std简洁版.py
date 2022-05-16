# _*_coding:utf-8_*_
# 2022-05-07
# windows 也可以运行

import os
import glob
import random
import shutil
import numpy as np
from PIL import Image
"""统计数据库中所图片的每个通道的 均值 和 标准差"""

if __name__ == '__main__':

    train_files = glob.glob(os.path.join('train','*','*.jpg'))

    print(f'Totally {len(train_files)} files for training')
    result = []

    for file in train_files:
        img = Image.open(file).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = img/255. # 注意，这里加了个点哎！
        result.append(img)

    print(np.shape(result)) #[BS, H, W, C]
    mean = np.mean(result,axis=(0,1,2))
    std = np.std(result,axis=(0,1,2)) #三个维度  那为什么不是 123呢？ batch_size维度 为什么被包含进去了啊？
    print(mean)
    print(std)

'''
Totally 1806 files for training
(1806, 128, 128, 3)
[0.46900262 0.44526845 0.33392775]
[0.37682414 0.36472037 0.35312928]
'''
