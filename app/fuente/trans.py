import cv2
import numpy

def dividirCanales(src):
    h,w,c = src.shape
    (B,G,R) = cv2.split(src)
    return B, R, G