#-*-coding:utf-8-*-

import cv2
import numpy as np

##############################图像去噪##################################
def img_filt(InputFile,OutputFile,OutputFile2): 
       #接口参数：(输入原图片，输出加噪声图片，输出已去噪图片)
    img = cv2.imread(InputFile,0) #直接读为灰度图像
    for i in range(2000): #添加2000个点噪声
        temp_x = np.random.randint(0,img.shape[0]) #x方向随机坐标
        temp_y = np.random.randint(0,img.shape[1]) #y方向随机坐标
        img[temp_x][temp_y] = 255 #随机点噪声
    blur = cv2.GaussianBlur(img,(5,5),0) #高斯滤波
    cv2.imwrite(OutputFile,img) #输出加噪声图片
    cv2.imwrite(OutputFile2,blur) #输出已去噪图片

##############################图像压缩##################################
def img_zip(InputFile, OutputFile):
    #接口参数：(输入原图片(jpg版)，输出已压缩图片)
    image = cv2.imread(InputFile) #读原图
    rows, cols, channels = image.shape #读原图尺寸
    res = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA)
    #使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
    cv2.imwrite(OutputFile, res,  [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    #输出已压缩图片，调整质量参数为50


############################图像边缘检测################################
def img_edgedetection(InputFile, OutputFile):
           #接口参数：(输入原图片，输出边缘图片)
    image = cv2.imread(InputFile) #读原图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #灰度化
    canny = cv2.Canny(gray, 30, 150) #30和150分别是高低阈值
    cv2.imwrite(OutputFile,canny) #输出边缘图片


##############################图像匹配##################################
def img_match(InputFile,InputFile2,OutputFile):
       #接口参数：(输入原图片，待匹配模板，输出已匹配图片)
    img_rgb = cv2.imread(InputFile) #读原图
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) #灰度化处理
    template = cv2.imread(InputFile2,0) #读模板
    w, h = template.shape[::-1] #读模板高度宽度
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) #匹配
    threshold = 0.7 #设定匹配阈值
    loc = np.where( res >= threshold) #找标记位置
    color = (0,0,255)       # 设定标记框颜色
    strokeWeight = 1        # 设定标记框粗细
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), color, strokeWeight)   #在原图中标记
    cv2.imwrite(OutputFile, img_rgb) #输出已匹配图片


##############################图像分割##################################
def img_segmentation(InputFile, OutputFile):
           #接口参数：(输入原图片，输出边缘图片)
    img = cv2.imread(InputFile) #读原图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度化
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #二值化
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2) #去噪
    
    sure_bg = cv2.dilate(opening,kernel,iterations=3) #背景区域

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0) #找前景区域

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg) #找未知区域

    ret, markers = cv2.connectedComponents(sure_fg) #标记

    markers = markers+1 #保证确定的背景区不是0是1

    markers[unknown==255]=0 #未知区域标0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0] #分割框颜色

    cv2.imwrite(OutputFile, img) #输出已分割图片


################################主函数##################################
if __name__ == '__main__':

    InputFile = "seg.jpg" 
    InputFile2 = "tmp.jpg"
    OutputFile = "out.jpg"
    OutputFile2 = "tmp2.jpg"

    #img_filt(InputFile, OutputFile, OutputFile2)
    #img_zip(InputFile, OutputFile)
    #img_edgedetection(InputFile, OutputFile)
    #img_match(InputFile, InputFile2, OutputFile)
    #img_segmentation(InputFile, OutputFile)

    print('done')