import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
# 图像反色
def image_opposite(image1_1):
    h, w = image1_1.shape
    image1_2 = image1_1.copy()
    for i in range(h):
        for j in range(w):
            image1_2[i, j] = 255 - image1_2[i, j]
    return image1_2

# 灰度化
def image2gray(img):
    # 获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]
    # 创建一幅图像
    grayimg = np.zeros((height, width), np.uint8)
    # 图像最大值灰度处理
    for i in range(height):
        for j in range(width):
            # 获取图像R G B最大值
            gray = max(img[i, j][0], img[i, j][1], img[i, j][2])
        # 灰度图像素赋值 gray=max(R,G,B)
            grayimg[i, j] = np.uint8(gray)
    return grayimg
# 读取原图片
image1 = cv2.imread("35652.jpg")
cv2.imshow("image1", image1)
cv2.waitKey(0)
cv2.destroyWindow('image1')
# 灰度化处理
# image1_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image1_1 = image2gray(image1)
print(image1_1.shape)
offset_X = 3
offset_Y = 5
# 切片提取内嵌区域
image1_1 = image1_1[offset_Y:-offset_Y, offset_X:-offset_X]
cv2.imshow("image1_1", image1_1)
cv2.waitKey(0)
cv2.destroyWindow('image1_1')
# 图像反色
image1_2 = image_opposite(image1_1)
cv2.imshow('image1_2', image1_2)
cv2.waitKey(0)
cv2.destroyWindow('image1_2')
# 图像二值化
#ret, image2 = cv2.threshold(image1_1,100,255, cv2.THRESH_BINARY)
image2 = cv2.adaptiveThreshold(image1_1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,15)
image2 = image_opposite(image2)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))  # 矩形结构
# image2 = cv2.dilate(image2, kernel)
cv2.imshow('image2', image2)
cv2.waitKey(0)
cv2.destroyWindow('image2')

# 水平投影
h1, w1= image2.shape  # 返回高和宽
image3 = image2.copy()
a = [0 for z in range(0, h1)]  # 初始化一个长度为w的数组，用于记录每一行的黑点个数
# 记录每一行的波峰
for j in range(0, h1):
    for i in range(0, w1):
        if image3[j, i] == 0:
            a[j] += 1
            image3[j, i] = 255

for j in range(0, h1):
    for i in range(0, a[j]):
        image3[j, i] = 0
cv2.waitKey(0)
plt.imshow(image3, cmap=plt.gray())  # 灰度图正确的表示方法
plt.show()
cv2.imshow('image3', image3)
cv2.waitKey(0)
# 垂直投影
image4 = image2.copy()
b = [0 for z in range(0, w2)]  # b = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数
# 记录每一列的波峰
for j in range(0, w2):  # 遍历一列
    for i in range(0, h2):  # 遍历一行
        if image4[i, j] == 0:  # 如果该点为黑点
            b[j] += 1  # 该列的计数器加一，最后统计出每一列的黑点个数
            image4[i, j] = 255  # 记录完后将其变为白色，相当于擦去原图黑色部分

for j in range(0, w2):
    for i in range((h2 - b[j]), h2):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
        image4[i, j] = 0  # 涂黑

plt.imshow(image4, cmap=plt.gray())
plt.show()
cv2.imshow('image4', image4)
cv2.waitKey(0)
# 分割字符
Position = []
start = 0
a_Start = []
a_End = []
# 根据水平投影获取垂直分割位置
for i in range(len(a)):
    if a[i] > 0 and start == 0:
        a_Start.append(i)
        start = 1
    if a[i] <= 0 and start == 1:
        a_End.append(i)
        start = 0
print(len(a_End))
# 分割行，分割之后再进行列分割并保存分割位置
for i in range(len(a_Start)):
    # 获取行图像
    cropImg = image2[a_Start[i]:a_End[i], 0:w1]
    # 对行图像进行垂直投影
    bstart = 0
    bend = 0
    b_Start = 0
    b_End = 0
    for j in range(len(b)):
        if b[j] > 0 and bstart == 0:
            b_Start = j
            bstart = 1
            bend = 0
        if b[j] <= 0 and bstart == 1:
            b_End = j
            bstart = 0
            bend = 1
        if bend == 1:
            Position.append([b_Start, a_Start[i], b_End, a_End[i]])
            bend = 0
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # 将灰度图转为RGB彩图

for m in range(len(Position)):
    if m!=2 :
        char_region = image2[Position[m][1]:Position[m][3], Position[m][0]:Position[m][2]]
        char_region = cv2.erode(char_region, kernel)
        cv2.imshow('rect', char_region)
        cv2.waitKey(0)
        cv2.imwrite('images/test/%i.jpg' % random.randint(1, 10000000), char_region)
# 根据确定的位置分割字符
for m in range(len(Position)):
    cv2.rectangle(image2, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 0, 255),2)
    # 第一个参数是原图；第二个参数是矩阵的左上点坐标；第三个参数是矩阵的右下点坐标；第四个参数是画线对应的rgb颜色；第五个参数是所画的线的宽度
cv2.imshow('rect', image2)
cv2.waitKey(0)