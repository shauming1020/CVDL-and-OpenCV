# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 11:40:19 2019

@author: DCMC
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from PyQt5.QtWidgets import QDialog, QApplication
from HW2_GUI import Ui_Form

import cv2 as cv
import numpy as np

class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.show()
        self.ui.btn1_1.clicked.connect(self.Q1_1)
        self.ui.btn2_1.clicked.connect(self.Q2_1)
        self.ui.btn3_1.clicked.connect(self.Q3_1)
        self.ui.btn3_2.clicked.connect(self.Q3_2)
        self.show()

    def Q1_1(self):
        
        imgL = cv.imread('imL.png',0)
        imgR = cv.imread('imR.png',0)
        
        cv.namedWindow('Left Image')
        cv.imshow('Left Image', imgL)
        cv.namedWindow('Right Image')
        cv.imshow('Right Image', imgR)

        stereo = cv.StereoBM_create(numDisparities=64, blockSize=9)
        disparity = stereo.compute(imgL,imgR)
        disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        cv.namedWindow('Q1_1_Disparity')
        cv.imshow('Q1_1_Disparity', disparity)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def Q2_1(self):
        
        img_rgb = cv.imread('ncc_img.jpg')
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        template = cv.imread('ncc_template.jpg',0)
        
        w, h = template.shape[::-1]
        result = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)

        loc = np.where(result >= 0.93)
        for pt in zip(*loc[::-1]):
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 2)

        cv.namedWindow('Q2_1_ncc_img')
        cv.imshow('Q2_1_ncc_img', img_rgb)
        cv.namedWindow('Q2_1_Template matching feature')
        cv.imshow('Q2_1_Template matching feature', result)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    def Q3_1(self):
        img1 = cv.imread('FeatureAerial1.jpg')
        self.img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.imread('FeatureAerial2.jpg')
        self.img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
        siftDetector = cv.xfeatures2d.SIFT_create(100)
        self.kp1, self.des1 = siftDetector.detectAndCompute(self.img1_gray, None)
        self.kp2, self.des2 = siftDetector.detectAndCompute(self.img2_gray, None)
        
        bf = cv.BFMatcher()
        matches = bf.knnMatch(self.des1, self.des2, k=2)

        self.good_pt = []
        for m, n in matches:
            if 0.4*n.distance < m.distance < 0.5*n.distance:
                self.good_pt.append(m)
        
        kp1_matched=([ self.kp1[m.queryIdx] for m in self.good_pt ])
        kp2_matched=([ self.kp2[m.trainIdx] for m in self.good_pt ]) 
            
        result1 = cv.drawKeypoints(self.img1_gray, kp1_matched, self.img1_gray)
        result2 = cv.drawKeypoints(self.img2_gray, kp2_matched, self.img2_gray)
        
        cv.namedWindow('Q3_1_FeatureAerial1')
        cv.imshow('Q3_1_FeatureAerial1', result1)
        cv.namedWindow('Q3_1_FeatureAerial2')
        cv.imshow('Q3_1_FeatureAerial2', result2)
        cv.namedWindow('Q3_1_FeatureAerial2')
        
        cv.imwrite('Q3_1_FeatureAerial1.jpg', result1)
        cv.imwrite('Q3_1_FeatureAerial2.jpg', result2)
        
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    def Q3_2(self):
        
        self.result = cv.drawMatchesKnn(self.img1_gray, self.kp1,\
                                    self.img2_gray, self.kp2,\
                                    [self.good_pt], None, flags=2) 
           
        cv.namedWindow('Q3_2_Result')
        cv.imshow('Q3_2_Result', self.result)  
        cv.imwrite('Q3_2_Result.jpg', self.result)
        
        cv.waitKey(0)
        cv.destroyAllWindows()
        
app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())