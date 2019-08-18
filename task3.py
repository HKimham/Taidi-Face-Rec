import cv2
import os
from os.path import join
import numpy as np
import random

resize_size = 64
npy_path = './faceImage_npy'
data = np.zeros((1,resize_size,resize_size))
labels = []
if not os.path.exists(npy_path):
    os.mkdir(npy_path)

for root, dirs, files in os.walk('./faceImageGray'):
    print(root)

    for file in files:
        img = cv2.imread(join(root,file),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(resize_size,resize_size),interpolation=cv2.INTER_LINEAR)
        img = img.reshape((1,resize_size,resize_size))
        data = np.concatenate((data, img), axis = 0)
        labels.append(root.split('\\')[-1])
    print(root.split('\\')[-1] + ' is done!')

data = np.delete(data,0,axis = 0)
labels = np.array(labels)

# 按相同方式打乱data和labels的顺序，后续直接按比例分割训练集和测试集
state = np.random.get_state()
np.random.shuffle(data)
np.random.set_state(state)
np.random.shuffle(labels)

print(data.shape)
print(labels.shape)

np.save(join(npy_path,'data.npy'), data)
print("save data.npy done")
np.save(join(npy_path,'labels.npy'), labels)
print("save labels.npy done")
