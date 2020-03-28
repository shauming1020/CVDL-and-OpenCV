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
        self.ui.btn4_1.clicked.connect(self.Q4_1)
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
        disparity = cv.normalize(disparity, disparity, alpha=0, beta=255,\
                                 norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        cv.namedWindow('Q1_1_Disparity')
        cv.imshow('Q1_1_Disparity', disparity)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def Q2_1(self):
        cap = cv.VideoCapture("bgSub.mp4")
        
        subtractor = cv.createBackgroundSubtractorMOG2(history=100,\
                                                       varThreshold=90,\
                                                       detectShadows=True)
        
        while True:
            _, frame = cap.read()
            mask = subtractor.apply(frame)
            cv.imshow("Q2_1_frame", frame)
            cv.imshow("Q2_1_mask", mask)
            key = cv.waitKey(30)
            if key == 27:
                break
        cap.release()
        cv.destroyAllWindows()
        
    def Q3_1(self):
        cap = cv.VideoCapture("featureTracking.mp4")
        _, frame = cap.read(0)
        
        params = cv.SimpleBlobDetector_Params()
 
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 100
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.82
         
        # Create a detector with the parameters
        detector = cv.SimpleBlobDetector_create(params)
    
        keypoints = detector.detect(frame)
        point2f = cv.KeyPoint_convert(keypoints)
        
        a = np.zeros([14, 2], dtype = np.int)
        
        for i in range(len(keypoints)):
            a[i][0] = int(point2f[i][0] - 4)
            a[i][1] = int(point2f[i][1] - 4)
           
            a[i+1][0] = int(point2f[i][0] + 4)
            a[i+1][1] = int(point2f[i][1] + 4)
        
            pt1, pt2 = tuple((a[i])), tuple((a[i + 1]))

            cv.rectangle(frame, pt1, pt2, (0, 0, 255), 1)
        
        cv.imshow("Q3_1", frame)
        cap.release()
        cv.waitKey(0)
        cv.destroyAllWindows()
        return
        
    def Q3_2(self):
        
        cap = cv.VideoCapture('featureTracking.mp4')     
        _, old_frame = cap.read()
        
        lk_params = dict(winSize  = (15, 15),\
                         maxLevel = 2,\
                         criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        params = cv.SimpleBlobDetector_Params()
         
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 100
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.82
        
        # Create a detector with the parameters
        detector = cv.SimpleBlobDetector_create(params)
        point2f = detector.detect(old_frame) 
        p0 = cv.KeyPoint_convert(point2f)
        
        # create an empty space temp to contain the feature points
        mask = np.zeros_like(old_frame)
        temp = np.zeros([14,2],dtype=np.int)
        
        for i in range(len(point2f)):
            temp[i][0] = int(p0[i][0] - 5)
            temp[i][1] = int(p0[i][1] - 5)
           
            temp[i + 1][0] = int(p0[i][0] + 5)
            temp[i + 1][1] = int(p0[i][1] + 5)
        
            pt1, pt2 = tuple((temp[i])), tuple((temp[i + 1]))
                
            cv.rectangle(old_frame, pt1, pt2,(0, 0, 255), 1)
        
        for picutre in range(320):
            _, frame = cap.read()
            
            if frame is None:
                break
            
            p1, st, err = cv.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
            
            i = 0
            for new, old in zip(p1, p0):
    
                np_1, np_2 = new.ravel()
                op_1, op_2 = old.ravel()
                cv.line(mask, (int(np_1), int(np_2)), (int(op_1), int(op_2)), (0, 255, 0), 1)
        
                temp[2 * i][0] = int(p1[i][0] - 5)
                temp[2 * i][1] = int(p1[i][1] - 5)
               
                temp[2 * i + 1][0] = int(p1[i][0] + 5)
                temp[2 * i + 1][1] = int(p1[i][1] + 5)
            
                tempx = tuple((temp[2 * i]))
                
                tempy = tuple((temp[2 * i + 1]))
                    
                cv.rectangle(mask,tempx,tempy,(0,0,255),1)
                i = i + 1
            
            img = cv.add(frame, mask)
            old_frame = frame.copy()
            p0 = p1
        
            cv.imshow('Q3_2.avi', img)
            
            k = cv.waitKey(30) 
            if k == 27:
                break
       
        cap.release()
        cv.destroyAllWindows()
      
        return
        
    def Q4_1(self):
        
        obj_p = np.zeros((8 * 11, 3), np.float32)
        obj_p[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1 , 2)
        
        obj_ps, img_ps, mtx, dist = [], [], [], []
        
        for i in range(1, 6):
            filename = './' + str(i) + '.bmp'
            img = cv.imread(filename)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            if ret:
                obj_ps.append(obj_p)
                img_ps.append(corners)
		
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_ps, img_ps, gray.shape[::-1], None, None)

        axis = np.float32([[1, 1, 0], [1, 5, 0], [5, 5, 0], [5, 1, 0], [3, 3, -4]])
		
        files, imgs = [], []
        for i in range(1, 6):
            files.append('./'+ str(i) + '.bmp')
        

        for h, filename in enumerate(files):
            img = cv.imread(filename)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            if ret:	
                imgpts, jac = cv.projectPoints(axis, rvecs[h], tvecs[h], mtx, dist)
				
                pt1 = tuple(imgpts[0].ravel())
                pt2 = tuple(imgpts[1].ravel())
                pt3 = tuple(imgpts[2].ravel())
                pt4 = tuple(imgpts[3].ravel())
                pt5 = tuple(imgpts[4].ravel())
				
                img = cv.line(img, pt1, pt2, (0, 0, 255), 10)
                img = cv.line(img, pt2, pt3, (0, 0, 255), 10)
                img = cv.line(img, pt3, pt4, (0, 0, 255), 10)
                img = cv.line(img, pt1, pt4, (0, 0, 255), 10)
                img = cv.line(img, pt1, pt5, (0, 0, 255), 10)
                img = cv.line(img, pt2, pt5, (0, 0, 255), 10)
                img = cv.line(img, pt3, pt5, (0, 0, 255), 10)
                img = cv.line(img, pt4, pt5, (0, 0, 255), 10)
                
                imgs.append(img)	
                height, width = img.shape[:2]

        output = cv.VideoWriter('Q4_1.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (width, height)) # 2 fps
        
        for i in range(len(imgs)):
            output.write(imgs[i])
        output.release()	
        
        cap = cv.VideoCapture('Q4_1.avi')
        if (cap.isOpened() == False):
            print('Already opened! ')
            
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                cv.namedWindow('Q4_1_Frame', cv.WINDOW_NORMAL)
                cv.imshow('Q4_1_Frame', frame)
                if cv.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                break
            
        cap.release()
        cv.destroyAllWindows()
        

app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())