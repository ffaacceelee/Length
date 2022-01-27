import numpy as np
from matplotlib import pyplot as plt
import cv2
import tkinter
import copy


# 轮廓面积计算函数
def areaCal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour[i])
    return area


def plot_demo(image):
    # numpy的ravel函数功能是将多维数组降为一维数组
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show("直方图")


def image_hist_demo(image):
    color = {"blue", "green", "red"}
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据，一般用在 for 循环当中。
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.subplot(3, 2, 6), plt.plot(hist, color=color)
        plt.subplot(3, 2, 6), plt.xlim([0, 256])
    # plt.show()


# INPUT IMAGE
img = cv2.imread('IMG.jpg')

# scale_percent = 60  # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# resize image

# RESIZE IMAGE
resized = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
# print(img)
# cv2.imshow('img',img)

# GAUSS AND GRAY
gauss = cv2.GaussianBlur(resized, (3, 3), 1)
imgray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)

# print(imgray)
# maxvalue = 255
# value = 3
# thresh = cv2.adaptiveThreshold(imgray, maxvalue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, value, 1)

# THRESH
ret, thresh = cv2.threshold(imgray, 127, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 127 is not important, OTSU is important
# print(thresh)

# CONTOURS
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
big_contour = max(contours, key=cv2.contourArea)  # FIND BIG ONE

# 绘制独立轮廓，如第四个轮廓
# imag = cv2.drawContour(img,contours,-1,(0,255,0),3)
# 但是大多数时候，下面方法更有用
# for c in range(len(contours)):

# DROW CONTOURS
imag = cv2.drawContours(resized, contours, -1, (255, 0, 0), 2, 8)

# HIST
hist = cv2.calcHist([imgray], [0], None, [256], [0, 256])

# CALCULATE
ar = [len(contours)]
le = [len(contours)]
ro = [len(contours)]
approx = [len(contours)]
aple = [len(contours)]
appr = [len(contours)]
for i in range(len(contours)):
    ar[i] = cv2.contourArea(contours[i])
    print("Area %d :" % i, end="")
    print(ar[i])
    le[i] = cv2.arcLength((contours[i]), True)
    print("arcLength %d :" % i, end="")
    print(le[i])
    ro[i] = ar[i] / le[i]
    print("ROUND %d :" % i, end="")
    print(ro[i])
    approx[i] = cv2.approxPolyDP(contours[i], le[i] * 0.05, True)
    aple[i] = cv2.arcLength((approx[i]), True)
    print("aple %d :" % i, end="")
    print(aple[i])
    appr[i] = le[i] / aple[i]
    print("appr %d :" % i, end="")
    print(appr[i])

resized2 = resized.copy()
imag2 = cv2.drawContours(resized2, approx, -1, (0, 255, 0), 2)

ellipse = cv2.fitEllipse(big_contour)
img2 = cv2.ellipse(resized2, ellipse, (0, 255, 255), 2)

print("long axis: %f" % ellipse[1][0])
print("short axis: %f" % ellipse[1][1])

st_x, st_y, width, height = cv2.boundingRect(big_contour)
print("x=", st_x)
print("y=", st_y)
print("w=", width)
print("h=", height)
bound_rect = np.array([[[st_x, st_y]], [[st_x + width, st_y]],
                       [[st_x + width, st_y + height]], [[st_x, st_y + height]]])

cv2.drawContours(img2, [bound_rect], -1, (255, 255, 255), 2)

# PLOT
# while (1):
plt.subplot(4, 2, 1), plt.imshow(resized)
plt.subplot(4, 2, 1), plt.title("resized")
plt.subplot(4, 2, 2), plt.imshow(gauss)
plt.subplot(4, 2, 2), plt.title("gauss")
plt.subplot(4, 2, 3), plt.imshow(imgray)
plt.subplot(4, 2, 3), plt.title("imgray")
plt.subplot(4, 2, 4), plt.imshow(thresh)
plt.subplot(4, 2, 4), plt.title("thresh")
plt.subplot(4, 2, 5), plt.imshow(imag)
plt.subplot(4, 2, 5), plt.title("imag")
plt.subplot(4, 2, 6), plt.plot(hist)
plt.subplot(4, 2, 6), plt.xlim([0, 255])
plt.subplot(4, 2, 6), plt.title("hist")
plt.subplot(4, 2, 7), plt.imshow(imag2)
plt.subplot(4, 2, 7), plt.title("imag2")
plt.show()

#     break
# cv2.destroyAllWindows()
