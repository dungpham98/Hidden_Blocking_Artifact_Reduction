import cv2
import numpy as np
path = "COCO_TEST.jpg"
img = cv2.imread(path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)