import setup_path 
import airsim

import cv2
import time
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base_module as bml


# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

def getTimeNow():
    return time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

def get_image_from_airsim(dst_dir='F:/Autodrive/airsim/airsimSorce/airsim_v1.20-1/PythonClient/tmp/',
                          name=None):

    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress = False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, pixels_as_float=True, compress = False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress = False),
        airsim.ImageRequest("0", airsim.ImageType.DepthVis, pixels_as_float=True, compress = False),
        airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized, pixels_as_float=True, compress = False),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, pixels_as_float=False, compress = False)
    ])

    i=-1
    for response in responses:
        i=i+1
        if not name:
            file_name = getTimeNow() + '_' + str(i) + '_DisparityNormalized'
        filename = dst_dir + file_name

        if response.pixels_as_float:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            f = airsim.get_pfm_array(response)
            bml.plot_picture([f], cmap = "gray", title=[str(i)])
            # airsim.write_pfm(os.path.normpath(filename + '.pfm'), f)
        elif response.compress: #png format
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            # airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        else: #uncompressed array
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #将图片的字符串数据变成一维数组
            img_rgba = img1d.reshape(response.height, response.width, 4) #reshape array to 4 channel image array H X W X 4
            bml.plot_picture([img_rgba], title=[str(i)])
            img_rgba = np.flipud(img_rgba) # 将图片数组的列向量元素顺序交换      
            # airsim.write_png(os.path.normpath(filename + '.png'), img_rgba) #write to png 

    plt.show()


def test_1():

    time.sleep(1)
    for i in range(1):
        get_image_from_airsim()

    # client.reset()
    client.enableApiControl(False)


def test_2():
    src = 'F:/Autodrive/airsim/airsimSorce/airsim_v1.20-1/PythonClient/'
    img1=mpimg.imread(src+'tmp/2018-07-20 11-09-39_0_DisparityNormalized')
    img2=mpimg.imread(src+'tmp/2018-07-19 17-13-22_3_DisparityNormalized.png')
    img3=cv2.imread(src+'tmp/2018-07-19 17-13-22_2_DisparityNormalized.png')
    img4=cv2.imread(src+'tmp/2018-07-19 17-13-22_3_DisparityNormalized.png')
    
    
    c = (img1 == img2)
    bml.plot_picture([img1],title=['test1.png'])
    bml.plot_picture([img2],title=['test2.png'])

    print(np.unique(c[:,:,0], return_counts=True)) #red
    print(np.unique(c[:,:,1], return_counts=True)) #green
    print(np.unique(c[:,:,2], return_counts=True)) #blue  
    print(np.unique(c[:,:,3], return_counts=True)) #blue
    plt.show()


test_2()