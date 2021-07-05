import cv2
import numpy as np


class DetectRect:
    def __init__(self):
        pass

    def detect(self, color_image):
        grayed = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(grayed, (7, 7), 0)
        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((2, 2), np.uint8)
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        thresh_img = cv2.threshold(gradient, 15, 225, cv2.THRESH_BINARY)[1]
        dilated = cv2.dilate(thresh_img, kernel2, iterations=2)
        # edged = cv2.Canny(gradient, 30, 40)
        (cnts, _) = cv2.findContours(dilated,
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
            if len(approx) == 4:
                Area = cv2.contourArea(cnt)
                if (Area > 30000):
                    (x, y, w, h) = cv2.boundingRect(cnt)
                else:
                    x = y = w = h = 0
            else:
                x = y = w = h = 0
