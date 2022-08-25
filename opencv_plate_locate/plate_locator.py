from opencv_plate_locate import util
import cv2 as cv
import numpy as np
import random
import os
def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


# 借助sobel算子完成车牌区域的提取
# 参数：车牌（含有车牌的车辆）图片
# 返回值：所有可能的车牌候选区域 = list

def get_candidate_plates(plate_image):
    # 1. 对含有车牌的车辆图片进行预处理(sobel)
    preprocess_image = util.preprocess_plate_image_by_sobel(plate_image)
    preprocess_image2 = util.preprocess_plate_image_by_hsv_blue(plate_image)
    # preprocess_image1 = util.preprocess_plate_image_by_hsv_green(plate_image)
    preprocess_image3 = cv.bitwise_and(preprocess_image2, preprocess_image)
    # cv.imshow('1',preprocess_image3)
    # cv.waitKey(0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (13,5))  # 矩形结构 blue
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (23, 7))
    preprocess_image3 = cv.dilate(preprocess_image3, kernel)
    cv.imshow('1', preprocess_image3)
    cv.waitKey(0)
    # 2. 提取所有的等值线|车牌轮廓(可能)的区域
    contours, _ = cv.findContours(preprocess_image3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # 3. 判断并获取所有的车牌区域的候选区域列表
    candidate_plates = []
    # 遍历所有的可能的车牌轮廓|等值线框
    # del_files('images/try/')
    for i in np.arange(len(contours)):
        # 逐一获取某一个可能的车牌轮廓区域
        contour = contours[i]
        output_image = util.rotate_plate_image(contour, plate_image)
        # cv.imshow('rect', output_image)
        # cv.waitKey(0)
        # cv.destroyWindow('rect')
        # 根据面积、长宽比判断是否是候选的车牌区域
        # if util.verify_plate_sizes(contour) and util.verify_plate_color(contour, plate_image):
        if util.verify_plate_sizes(contour):
            # 完成旋转
            output_image = util.rotate_plate_image(contour, plate_image)
            # 统一尺寸
            uniformed_image = util.unify_plate_image(output_image)
            cv.imwrite('images/try/%i.jpg'%random.randint(100000,1000000), uniformed_image)
            cv.imshow('rect', uniformed_image)
            cv.waitKey(0)
            cv.destroyWindow('rect')
            # 追加到车牌区域的候选区域列表中
            candidate_plates.append(uniformed_image)

    # 返回含有所有的可能车牌区域的候选区域列表
    return candidate_plates
    # return contours


def get_candidate_plates_video(plate_image):
    # 1. 对含有车牌的车辆图片进行预处理(sobel)
    preprocess_image = util.preprocess_plate_image_by_sobel(plate_image)
    preprocess_image2 = util.preprocess_plate_image_by_hsv_blue(plate_image)
    # preprocess_image1 = util.preprocess_plate_image_by_hsv_green(plate_image)
    preprocess_image3 = cv.bitwise_and(preprocess_image2, preprocess_image)
    # cv.imshow('1',preprocess_image3)
    # cv.waitKey(0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,3))  # 矩形结构 blue
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (23, 7))
    preprocess_image3 = cv.dilate(preprocess_image3, kernel)
    # cv.imshow('1', preprocess_image3)
    # cv.waitKey(0)
    # 2. 提取所有的等值线|车牌轮廓(可能)的区域
    contours, _ = cv.findContours(preprocess_image3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # 3. 判断并获取所有的车牌区域的候选区域列表
    candidate_plates = []
    contours1 = []
    # 遍历所有的可能的车牌轮廓|等值线框
    # del_files('images/try/')
    for i in np.arange(len(contours)):
        # 逐一获取某一个可能的车牌轮廓区域
        contour = contours[i]
        output_image = util.rotate_plate_image(contour, plate_image)
        # cv.imshow('rect', output_image)
        # cv.waitKey(0)
        # cv.destroyWindow('rect')
        # 根据面积、长宽比判断是否是候选的车牌区域
        # if util.verify_plate_sizes(contour) and util.verify_plate_color(contour, plate_image):
        if util.verify_plate_sizes_video(contour):
            # 完成旋转
            output_image = util.rotate_plate_image(contour, plate_image)
            # 统一尺寸
            uniformed_image = util.unify_plate_image(output_image)
            cv.imwrite('../images/try/%i.jpg'%random.randint(100000,1000000), uniformed_image)
            # cv.imshow('rect', uniformed_image)
            # cv.waitKey(0)
            # cv.destroyWindow('rect')
            # 追加到车牌区域的候选区域列表中
            candidate_plates.append(uniformed_image)
            contours1.append(contour)

    # 返回含有所有的可能车牌区域的候选区域列表
    # return candidate_plates
    return contours1


