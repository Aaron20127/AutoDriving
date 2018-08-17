# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:37:10 2017

@author: yang
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lane_line_detection_utils as utils
import base_module as bml


def thresholding(img):
#    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=55, thresh_max=100)
#    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(70, 255))
#    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
#    s_thresh = utils.hls_select(img,channel='s',thresh=(160, 255))
#    s_thresh_2 = utils.hls_select(img,channel='s',thresh=(200, 240))
#    
#    white_mask = utils.select_white(img)
#    yellow_mask = utils.select_yellow(img)

    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=10 ,thresh_max=230)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = utils.hls_select(img, thresh=(180, 255))
    lab_thresh = utils.lab_select(img, thresh=(155, 200))
    luv_thresh = utils.luv_select(img, thresh=(225, 255))
    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1

    return threshholded

def processing(img,object_points,img_points,M,Minv,left_line,right_line):
    # undist = utils.cal_undistort(img,object_points,img_points)
    undist = img
    thresholded = thresholding(undist)
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = \
        utils.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)
    
    # bml.plot_picture([undist, thresholded, thresholded_wraped],
    #                  title=['undist', 'thresholded', 'thresholded_wraped'])
    # plt.show()

    area_img = utils.draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)
    
    curvature,pos_from_center = utils.calculate_curv_and_pos(thresholded_wraped,left_fit, right_fit)
    result = utils.draw_values(area_img,curvature,pos_from_center)
    return result

def test_1():
    img=mpimg.imread('lane_line_detection/CarND-Advanced-Lane-Lines/test_images/test2.png')
    bml.plot_picture([img],title=['test1.png'])

    ### 1.多种混合方案实现阈值过滤
    thresh = thresholding(img)
    bml.plot_picture([thresh], 'gray', title=['thresholding']) 

    ### 2.平面图转换成鸟瞰图
    M, Minv = utils.get_M_Minv()
    thresholded = thresholding(img)
    thresholded_wraped_img = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    bml.plot_picture([thresholded_wraped_img], 'gray', title=['thresholded_wraped_img'])
    bml.plot_picture([thresholded_wraped], 'gray', title=['thresholded_wraped'])

    ### 3.灰度直方图
    bml.plot_base([np.sum(thresholded_wraped, axis=0)],title = 'histogram') 

    plt.show()

def test_2():
    src_base = 'lane_line_detection/CarND-Advanced-Lane-Lines/test_images/'
    # src = 'lane_line_detection/CarND-Advanced-Lane-Lines/test_images/test_1.png'
    src = src_base + 'mytest.png'
    img=cv2.imread(src)

    # img=plt.imread(src)
    # img=img * 255 
    # img = img.astype(np.uint8)

    # img = img / 255.0
    # img1=plt.imread(src) 
    # bml.plot_picture([img],title=['img.png'])
    # bml.plot_picture([img1],title=['img1.png'])

    # src = 'lane_line_detection/CarND-Advanced-Lane-Lines/test_images/test1.jpg'
    # img=cv2.imread(src)
    # img = img / 255.0
    # img1=plt.imread(src) 
    # bml.plot_picture([img],title=['img.jpg'])
    # bml.plot_picture([img1],title=['img1.jpg'])

    # plt.show()

    left_line = utils.Line()
    right_line = utils.Line()
    cal_imgs = utils.get_images_by_dir('lane_line_detection/CarND-Advanced-Lane-Lines/camera_cal')
    object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))
    M,Minv = utils.get_M_Minv()

    img_processing = processing(img, object_points,img_points, M, Minv, left_line, right_line)
    # bml.plot_picture([img_processing],title=['img_processing'])
    # plt.show()
    cv2.imwrite(src_base + 'my_75d_process.png', img_processing)
    cv2.namedWindow('img_processing')
    cv2.imshow('img_processing',img_processing)
    cv2.waitKey()
    cv2.destroyAllWindows()

def test3():
    # 底板图案
    bottom_pic = 'lane_line_detection/CarND-Advanced-Lane-Lines/test_images/test1.png'
    # 上层图案
    top_pic = 'lane_line_detection/CarND-Advanced-Lane-Lines/test_images/test2.png'

    import cv2
    bottom = cv2.imread(bottom_pic)
    top = cv2.imread(top_pic)
    # 权重越大，透明度越低
    overlapping = cv2.addWeighted(bottom, 0.8, top, 0.2, 0)
    # 保存叠加后的图片
    cv2.imwrite('overlap(8:2).jpg', overlapping)

    cv2.namedWindow('addImage')
    cv2.imshow('img_add',overlapping)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_2()

# project_outpath = 'vedio_out/project_video_out.mp4'
# project_video_clip = VideoFileClip("project_video.mp4")
# project_video_out_clip = project_video_clip.fl_image(
#     lambda clip: processing(clip,object_points,img_points,M,Minv,left_line,right_line))
# project_video_out_clip.write_videofile(project_outpath, audio=False)



