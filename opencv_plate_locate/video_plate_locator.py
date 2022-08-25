import numpy as np
import cv2 as cv
import opencv_plate_locate.plate_locator as pl


def video_plate_locate_1(path='../video/daolu1.avi'):
    # 1.获取图像
    cap = cv.VideoCapture(path)
    # 2. 目标追踪
    # 2.1 设置窗口搜索终止条件：最大迭代次数，窗口中心漂移最小值
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    flag = 0
    # while(True):
    #     # 2.2 获取每一帧图像
    #     ret, frame = cap.read()
    #     if(flag%5 == 0):
    #         contours = pl.get_candidate_plates_video(frame)
    #         track_window = []
    #         roi_hist = []
    #         for i in range(len(contours)):
    #             c, r, w, h = cv.boundingRect(contours[i])
    #             track_window.append([c, r, w, h])
    #             # print(frame.shape, track_window[i])
    #             roi = frame[r:r + h, c:c + w]
    #             hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    #             roi_hist.append(cv.calcHist([hsv_roi], [0], None, [180], [0, 180]))
    #             cv.normalize(roi_hist[i], roi_hist[i], 0, 255, cv.NORM_MINMAX)
    #     if ret == True:
    #         for i in range(len(contours)):
    #             # 4.3 计算直方图的反向投影
    #             hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #             dst = cv.calcBackProject([hsv], [0], roi_hist[i], [0, 180], 1)
    #             # 4.4 进行meanshift追踪
    #             ret, track_window[i] = cv.meanShift(dst, track_window[i], term_crit)
    #             # 4.5 将追踪的位置绘制在视频上，并进行显示
    #             x, y, w, h = track_window[i]
    #             frame = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
    #         cv.imshow('frame',frame)
    #         flag+=1
    #         print(flag)
    #         if cv.waitKey(200) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    # # 3. 资源释放
    # cap.release()
    # cv.destroyAllWindows()
    while (True):
        # 2.2 获取每一帧图像
        ret, frame = cap.read()
        contours = pl.get_candidate_plates_video(frame)
        track_window = []
        roi_hist = []
        for i in range(len(contours)):
            c, r, w, h = cv.boundingRect(contours[i])
            track_window.append([c, r, w, h])
            # print(frame.shape, track_window[i])
            roi = frame[r:r + h, c:c + w]
            hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            roi_hist.append(cv.calcHist([hsv_roi], [0], None, [180], [0, 180]))
            cv.normalize(roi_hist[i], roi_hist[i], 0, 255, cv.NORM_MINMAX)

        if ret == True:
            for i in range(len(contours)):
                # 4.3 计算直方图的反向投影
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist[i], [0, 180], 1)
                # 4.4 进行meanshift追踪
                ret, track_window[i] = cv.meanShift(dst, track_window[i], term_crit)
                # 4.5 将追踪的位置绘制在视频上，并进行显示
                x, y, w, h = track_window[i]
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv.imshow('frame', frame)
            flag += 1
            print(flag)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # 3. 资源释放
    cap.release()
    cv.destroyAllWindows()


def video_plate_locate_2(path='../video/daolu1.avi'):
    # 1.获取图像
    cap = cv.VideoCapture(path)
    # 2. 目标追踪
    # 2.1 设置窗口搜索终止条件：最大迭代次数，窗口中心漂移最小值
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 1)
    flag = 0
    while(True):
        # 2.2 获取每一帧图像
        ret, frame = cap.read()
        if(flag%30 == 0):
            contours = pl.get_candidate_plates_video(frame)
            track_window = []
            roi_hist = []
            for i in range(len(contours)):
                c, r, w, h = cv.boundingRect(contours[i])
                track_window.append([c, r, w, h])
                # print(frame.shape, track_window[i])
                roi = frame[r:r + h, c:c + w]
                hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
                roi_hist.append(cv.calcHist([hsv_roi], [0], None, [180], [0, 180]))
                cv.normalize(roi_hist[i], roi_hist[i], 0, 255, cv.NORM_MINMAX)
        if ret == True:
            for i in range(len(contours)):
                # 4.3 计算直方图的反向投影
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist[i], [0, 180], 1)
                # 4.4 进行meanshift追踪
                ret, track_window[i] = cv.meanShift(dst, track_window[i], term_crit)
                # 4.5 将追踪的位置绘制在视频上，并进行显示
                x, y, w, h = track_window[i]
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv.imshow('frame',frame)
            flag+=1
            print(flag)
            if cv.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break
    # 3. 资源释放
    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    video_plate_locate_1()
    # video_plate_locate_2()