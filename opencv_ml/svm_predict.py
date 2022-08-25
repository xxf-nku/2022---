import os
import numpy as np
import cv2 as cv
from sklearn import svm

#31行改为测试好图片的路径（该图片已经分割完毕，并且按顺序排列），然后直接运行

import joblib
from opencv_ml import ml_predict_utility





import joblib
from opencv_ml import ml_predict_utility


def svm_predict(TEST_DIR = "images/test"):
	ENGLISH_LABELS = [
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
		'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
		'W', 'X', 'Y', 'Z']

	CHINESE_LABELS = [
		"川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津",
		"京", "吉", "辽", "鲁", "蒙", "闽", "宁", "青", "琼", "陕",
		"苏", "晋", "皖", "湘", "新", "豫", "渝", "粤", "云", "藏",
		"浙"]

	ENGLISH_MODEL_PATH = "./model/svm/svm_enu.m"
	CHINESE_MODEL_PATH = "./model/svm/svm_chs.m"
	ENGLISH_IMAGE_WIDTH = 20
	ENGLISH_IMAGE_HEIGHT = 20
	CHINESE_IMAGE_WIDTH = 24
	CHINESE_IMAGE_HEIGHT = 48
	# 路径
	TEST_DIR = "./images/test"

	# 获取该路径下全部图片路径
	def load_path(dir_path):
		PATH = []
		# 获取数据集目录下的所有的子目录，并逐一遍历
		for item in os.listdir(dir_path):
			# 获取每一个具体样本类型的 os 的路径形式
			item_path = os.path.join(dir_path, item)
			PATH.append(item_path)
			if os.path.isdir(item_path):
				PATH.append(item_path)
		return PATH

	ans = []

	testimages = load_path(TEST_DIR)
	# 第一个汉字识别输出
	image_path = testimages[0]

	chinese_image = ml_predict_utility.load_image(image_path, CHINESE_IMAGE_WIDTH, CHINESE_IMAGE_HEIGHT)
	chs_model = joblib.load(CHINESE_MODEL_PATH)
	predicts = chs_model.predict(chinese_image)
	# print(CHINESE_LABELS[predicts[0]])
	ans.append(CHINESE_LABELS[predicts[0]])

	# 之后数字和字母识别输出
	for testimage in testimages:
		if testimage == testimages[0]:
			continue
		image_path = testimage
		english_image = ml_predict_utility.load_image(image_path, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)
		enu_model = joblib.load(ENGLISH_MODEL_PATH)
		chs_model = joblib.load(CHINESE_MODEL_PATH)
		predicts = enu_model.predict(english_image)
		ans.append(ENGLISH_LABELS[predicts[0]])
	# print(ENGLISH_LABELS[predicts[0]])

	content = "".join(ans)
	print(content)
	return content




