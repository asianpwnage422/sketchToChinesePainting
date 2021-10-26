import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology

IMG_PATH = "/Users/raisemmina/Desktop/sketchToChinesePainting/dataset_gray"
SAVE_PATH = "/Users/raisemmina/Desktop/sketchToChinesePainting/thin"

def binarization(img):
    ret,img = cv2.threshold(img,160,255, cv2.THRESH_BINARY)
    img[np.where(img <= 127)] = 1
    img[np.where(img > 127)] = 0
    return img

def de_binary(img):
    img=255-img
    return img

def load_data(img_path):
    #size = 109
    img = Image.open(img_path)
    #img = img.resize((size, size), Image.BILINEAR)
    gray = img.convert('L')
    gray = np.array(gray)
    #kernel_size = 3
    blur_gray = cv2.bilateralFilter(gray,7, sigmaSpace = 75, sigmaColor =75)
    blur_gray = np.array(blur_gray)
    blur_gray = binarization(blur_gray)
    return blur_gray

def thin(img):
    #skel, distance = morphology.medial_axis(img, return_distance=True)
    #dist_on_skel = distance * skel
    #dist_on_skel = dist_on_skel.astype(np.uint8)*255
    img = morphology.skeletonize(img).astype(np.uint8)*255
    return img

def dil_ero(img):
    kernel = np.ones((5,5),np.uint8)
    #erosion = cv2.erode(img, kernel, iterations = 1)
    img = cv2.dilate(img, kernel, iterations = 1)
    #img = cv2.erode(img, kernel, iterations = 1)
    return img
i=1
for root, dirs, fs in os.walk(IMG_PATH):
    for f in fs:
            p = os.path.join(root, f)
            img = load_data(p)
            img = thin(img)
            img = dil_ero(img)
            img = de_binary(img)
            save_name = '%s.png' % (f[:-4])
            i=i+1
            print(save_name)
            cv2.imencode('.png', img)[1].tofile('%s/' % SAVE_PATH + save_name)

