#-*-coding:utf-8-*-

import cv2
import numpy as np

##############################图像去噪##################################
def img_filt(InputFile, OutputFile):
    img = cv2.imread(InputFile) #直接读为灰度图像
    img1 = np.float32(img) #转化数值类型
    kernel = np.ones((5,5),np.float32)/25

    dst = cv2.filter2D(img1,-1,kernel)

    cv2.imwrite(OutputFile, dst)


##############################图像压缩##################################
def img_zip(InputFile, OutputFile):
    image = cv2.imread(InputFile)
    rows, cols, channels = image.shape
    res = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA)
    cv2.imwrite(OutputFile, res)


############################图像边缘检测################################
def img_edgedetection(InputFile, OutputFile):
    image = cv2.imread(InputFile)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #30 and 150 is the threshold, larger than 150 is considered as edge,
    #less than 30 is considered as not edge
    canny = cv2.Canny(gray, 30, 150)
    #display two images in a figure
    cv2.imwrite(OutputFile,canny)


##############################图像匹配##################################
def img_match(InputFile,InputFile2,OutputFile):
    img_rgb = cv2.imread(InputFile)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(InputFile2,0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    # rectangle color and stroke
    color = (0,0,255)       # reverse of RGB (B,G,R) - weird
    strokeWeight = 1        # thickness of outline
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), color, strokeWeight)   
    cv2.imwrite(OutputFile, img_rgb)


##############################图像分割##################################
def img_segmentation(InputFile, OutputFile):
    img = cv2.imread(InputFile)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255]=0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    cv2.imwrite(OutputFile, img)


################################主函数##################################
if __name__ == '__main__':

    InputFile = "seg.jpg" 
    InputFile2 = "tmp.jpg"
    OutputFile = "out5.jpg"

    img_filt(InputFile, OutputFile)
    #img_zip(InputFile, OutputFile)
    #img_edgedetection(InputFile, OutputFile)
    #img_match(InputFile, InputFile2, OutputFile)
    #img_segmentation(InputFile, OutputFile)

    print('done')