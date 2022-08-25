import cv2 as cv
import numpy as np
import os

def normalize_data(data):
    return (data - data.mean()) / data.max()

def load_image(image_path, width, height):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    resized_image = cv.resize(gray_image, (width, height))
    normalized_image = normalize_data(resized_image)
    data = []
    data.append(normalized_image.ravel())
    return np.array(data)


# 获取该路径下全部图片路径
def load_path(dir_path):
    path = []
    for item in os.listdir(dir_path):
        # 获取每一个具体样本类型的 os 的路径形式
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            path1 = load_path(item_path)
            #path.append(item_path)
            path = path + path1
            #print(item_path)
        else:
            # print(item_path)
            path.append(item_path)
    return path
