# 导包

import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)
# 获取车牌区域的字符拆分后的候选字符列表
# 参数：车牌候选区域（某一个）， candidate_plate_image
# 返回值：切分车牌字符，按顺序生成的车牌字符候选列表


# 获取车牌区域的字符拆分后的候选字符列表
# 参数：车牌候选区域（某一个）， candidate_plate_image
# 返回值：切分车牌字符，按顺序生成的车牌字符候选列表
def get_candidate_chars(candidate_plate_image):
    # 1. 图片预处理：非白色涂为蓝色+灰度+二值化
    # gray_image = cv.cvtColor(candidate_plate_image, cv.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(candidate_plate_image, cv2.COLOR_BGR2GRAY)
    is_success, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)

    # 2. 向内缩进，去除外边框
    # 经验值
    offset_X = 3
    offset_Y = 5
    # 切片提取内嵌区域
    offset_region = binary_image[offset_Y:-offset_Y, offset_X:-offset_X]
    # 生成工作区域
    working_region = offset_region

    # 3. 对车牌区域进行等值线找区域（要先处理下汉字-模糊化）
    # 经验值：汉字区域占整体的 1/8
    chinese_char_max_width = working_region.shape[1] // 8
    # 提取汉字区域
    chinese_char_region = working_region[:, 0:chinese_char_max_width]
    # 对汉字区域进行模糊处理
    cv2.GaussianBlur(chinese_char_region, (9, 9), 0, dst=chinese_char_region)
    # 对整个区域找轮廓==等值线
    char_contours, _ = cv2.findContours(working_region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # 4. 过滤不合适的轮廓（等值线框）
    # 经验值
    CHAR_MIN_WIDTH  = working_region.shape[1] // 25
    CHAR_MIN_HEIGHT = working_region.shape[0] * 7 // 10
    CHAR_MAX_WIDIH = working_region.shape[1] // 7
    CHAR_MAX_HEIGHT = working_region.shape[0] * 20 // 10

    # 5. 逐个遍历所有候选的字符区域轮廓==等值线框，按照大小进行过滤
    valid_char_regions = []
    for i in np.arange(len(char_contours)):
        x, y, w, h = cv2.boundingRect(char_contours[i])
        if w >= CHAR_MIN_WIDTH and h >= CHAR_MIN_HEIGHT and w <= CHAR_MAX_WIDIH and h <= CHAR_MAX_HEIGHT :
            # 将字符区域的中心点x的坐标 和 字符区域 作为一个元组，放入 valid_char_regions 列表
            valid_char_regions.append((x, offset_region[y:y + h, x:x + w]))

    # 6. 按找区域的x坐标进行排序，并返回字符列表
    sorted_regions = sorted(valid_char_regions, key=lambda region: region[0])
    # valid_char_regions
    # sorted_regions
    candidate_char_images = []
    # 清空文件夹
    del_files("images/test/")
    for i in np.arange(len(sorted_regions)):
        candidate_char_images.append(sorted_regions[i][1])
        cv.imwrite('images/test/%i.jpg' %random.randint(1,10000000), sorted_regions[i][1])

    return candidate_char_images

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
            gray = 0.00 * img[i,j][0] + 0.00 * img[i,j][1] + 1* img[i,j][2]
        # 灰度图像素赋值 gray=max(R,G,B)
            grayimg[i, j] = np.uint8(gray)
    return grayimg


# def find_realPosition(position):
#     real_position = []
#     k = position[0][0]
#     for i in (len(position)-1):
#         if position[i][2] - k > 9:
#             real_position.append(k, position[i][1], position[i][2], position[i][3])
#             # k = position[i][2]
#         else:
#             if position[i][2] - position[i][0] < 9:
#                 k = position[i][0]
#             else:
#                 real_position.append(position[i][0], position[i][1], position[i][2], position[i][3])



def get_candidate_char(candidate_plate_image):
    # 读取原图片
    image1 = candidate_plate_image.copy()
    #cv2.fastNlMeansDenoisingColored()
    # cv2.imshow("image1", image1)
    # cv2.waitKey(0)
    # cv2.destroyWindow('image1')
    # 2. 向内缩进，去除外边框
    # 经验值
    offset_X = 1
    offset_Y = 1
    # 切片提取内嵌区域
    offset_region = image1[offset_Y:-offset_Y, offset_X:-offset_X]
    # 生成工作区域
    image1 = offset_region
    # 灰度化处理
    #image1_1 = image2gray(image1)
    image1_1 = cv2.cvtColor(image1,  cv2.COLOR_BGR2GRAY)
    # image1_1 = cv2.fastnlmeansdenosing(image1_1)
    # image1_1 = cv2.GaussianBlur(image1_1, (3, 1), 0)
    # cv2.imshow('image1_1', image1_1)
    # cv2.waitKey(0)
    # cv2.destroyWindow('image1_1')
    #图像二值化
    ret, image2 = cv2.threshold(image1_1, 50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #image2 = cv2.adaptiveThreshold(image1_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))  # 矩形结构
    image2 = cv2.erode(image2, kernel)
    image2 = cv2.dilate(image2, kernel)
    image2 = image_opposite(image2)
    # image2 = cv2.bilateralFilter(image2, 9, 150, 150)
    # cv2.imshow('image2', image2)
    # cv2.waitKey(0)
    # cv2.destroyWindow('image2')
    chinese_char_max_width = image2.shape[1] // 8 + 5
    chinese_char_region = image2[:, 0:chinese_char_max_width]
    chinese_char_region = image_opposite(chinese_char_region)
    del_files('images/test/')
    #cv2.imwrite('images/test/province.jpg', chinese_char_region)
    cv2.imwrite('images/test/0.jpg', chinese_char_region)
    image2 = image2[:, chinese_char_max_width:]
    # kernel = np.ones((2, 2), np.uint8)
    # image2 = cv2.erode(image2, kernel)
    # cv2.imshow('image2', image2)
    # cv2.waitKey(0)
    # cv2.destroyWindow('image2')
    # 水平投影
    h1, w1 = image2.shape  # 返回高和宽
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
    plt.imshow(image3, cmap=plt.gray())  # 灰度图正确的表示方法
    plt.imshow(image3, cmap=plt.gray())
    plt.show()
    cv2.imshow('image3', image3)
    cv2.waitKey(0)
    cv2.destroyWindow('image3')
    # 垂直投影
    h2, w2 = image2.shape  # 返回高和宽
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
    cv2.destroyWindow('image4')
    # 分割字符
    # 查找水平直方图波峰
    # print(len(a))
    # x_histogram = np.sum(image3, axis=1)
    x_min = np.percentile(a, 35)
    x_max = np.max(a)
    x_average = np.sum(a) /len(a)
    x_threshold = np.percentile(a, 20)
    #x_threshold = (x_min+x_average)/2
    # 查找垂直直方图波峰
    # row_num, col_num = image2.shape[:2]
    # # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
    # gray_img = image2[1:row_num - 1]
    y_min = np.min(b)
    y_threshold = np.percentile(b,20)  # U和0要求阈值偏小，否则U和0会被分成两半
    Position = []
    start = 0
    a_Start = []
    a_End = []
    # 根据水平投影获取垂直分割位置
    for i in range(len(a)):
        if  start == 0 and a[i] >= x_min :
            a_Start.append(i)
            start = 1
            continue
        if a[i] < x_threshold and start == 1 :
            a_End.append(i)
            start = 0
    # print(a_Start)
    # print(a_End)
    # 分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(a_End)):
        # 获取行图像
        cropImg = image2[a_Start[i]:a_End[i], 0:w1]
        # 对行图像进行垂直投影
        bstart = 0
        bend = 0
        b_Start = 0
        b_End = 0
        for j in range(len(b)):
            if b[j] > y_min and bstart == 0:
                b_Start = j
                bstart = 1
                bend = 0
                continue
            if b[j] <= y_threshold and bstart == 1:
                b_End = j
                bstart = 0
                bend = 1
            if bend == 1:
                Position.append([b_Start, a_Start[i], b_End, a_End[i]])
                bend = 0
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # 将灰度图转为RGB彩图
    print(Position)
    for m in range(len(Position)):
        if Position[m][2]-Position[m][0] > 4 and Position[m][3] - Position[m][1] > 5 :
            char_region = image2[Position[m][1]:Position[m][3], Position[m][0]:Position[m][2]]
            #
            #char_region = cv2.dilate(char_region, kernel)
            char_region = image_opposite(char_region)
            # cv2.imshow('rect', char_region)
            # cv2.waitKey(0)
            cv2.imwrite('images/test/%i.jpg' % (m+1), char_region)
    # 根据确定的位置分割字符
    for m in range(len(Position)):
        cv2.rectangle(image2, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 0, 255), 2)
        # 第一个参数是原图；第二个参数是矩阵的左上点坐标；第三个参数是矩阵的右下点坐标；第四个参数是画线对应的rgb颜色；第五个参数是所画的线的宽度
    # cv2.imshow('rect', image2)
    # cv2.waitKey(0)
    # cv2.destroyWindow('rect')


if __name__ == '__main__':
    get_candidate_char()


