import warnings
warnings.filterwarnings("ignore")
import opencv_char_seperator.plate_char_seperator as ocs
import opencv_plate_locate.plate_locator as pl
import opencv_ml.svm_predict as  svm_predict
import cv2
import opencv_ml.ml_predict_utility as ml_predict_utility
import test_LPRNet as lpr
from opencv_plate_locate import util

def text_save(filename, data):
    #filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    file.truncate();
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def main_images(path="images/.ipynb_checkpoints"):
    image_paths = ml_predict_utility.load_path(path)
    plates_images = []
    str = []
    for image_path in image_paths:
        plates_images.append(cv2.imread(image_path))
    for i in range(len(plates_images)):
        candidate_plates = pl.get_candidate_plates(plates_images[i])
        print(i)
        # for j in range(len(candidate_plates)):
        #      # ocs.get_candidate_char(candidate_plates[j])
        #     # svm_predict.svm_predict()会返回一个字符串, 为识别的结果
        #
        #      str.append(svm_predict.svm_predict())
        # text_save('result/myresult', str)
    lpr.test('images/try/')


def main_image(path='images/plate2.jpg'):
    plate_image = cv2.imread(path)  # 获取车牌图片
    str = []
    candidate_plates = pl.get_candidate_plates(plate_image)
    for j in range(len(candidate_plates)):
        ocs.get_candidate_char(candidate_plates[j])
        str.append(svm_predict.svm_predict())
    text_save('result/myresult', str)
    # lpr.test('images/try/')
    return str


def main_video(path):
    return 0
if __name__ == '__main__':
    # main_images('images/test2')
    main_image('images/plate3.jpg')
    # lpr.test('./images/try/150041.jpg')
    # img = cv2.imread('images/try/125050.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # is_success, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # img = util.unify_plate_image(img)
    # cv2.imshow('1',img)
    # cv2.waitKey(0)




# 效果好的为images目录下的1，3，4jpg



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

