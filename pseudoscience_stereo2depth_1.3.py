# Josh Gladstone made this in 2018
# But a lot of it came from Timotheos Samartzidis
# http://timosam.com/python_opencv_depthimage
# Enjoy!

import numpy as np
import cv2
import sys, os
 
print('Loading images...')
numberofimages = len(sys.argv)-1
if (numberofimages > 1):
    print ('Batch processing ' + str(numberofimages) + ' images')
    pathname = os.path.dirname(sys.argv[1]) + '\\batch'
else:
    pathname = os.path.dirname(sys.argv[1])
filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]
imgg = cv2.imread(sys.argv[1])
heightg, widthg = imgg.shape[:2]
imggL = imgg[0:int((heightg/2)), 0:widthg]
imggR = imgg[int((heightg/2)):heightg, 0:widthg]

def Nothing(value):
    if (shouldirun == 1):
        Update(0)
    else:
        return None

def Update(value):
    if (value == 0):
        img = cv2.imread(sys.argv[1])
    else:
        img = cv2.imread(sys.argv[value])
    height, width = img.shape[:2]
    imgL = img[0:int((height/2)), 0:width]
    imgR = img[int((height/2)):height, 0:width]
    minDisparities=16
    window_size = cv2.getTrackbarPos('windowSize','settings')                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    rez = cv2.getTrackbarPos('resolution','settings')/20.0
    if (rez > 0):
        resL = cv2.resize(imgL,None,fx=rez, fy=rez, interpolation = cv2.INTER_AREA)
        resR = cv2.resize(imgR,None,fx=rez, fy=rez, interpolation = cv2.INTER_AREA)

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=minDisparities,             # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize= cv2.getTrackbarPos('blockSize','settings'),
            P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap= cv2.getTrackbarPos('FilterCap','settings'),
            mode=cv2.STEREO_SGBM_MODE_HH
        )
         
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
         
        # FILTER Parameters
        lmbda = cv2.getTrackbarPos('lmbda','settings') * 1000
        sigma = 1.2
        visual_multiplier = 1
         
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        if (value == 0): 
            print('Calculating depth map')
        displ = left_matcher.compute(resL, resR)
        dispr = right_matcher.compute(resR, resL)
        imgLb = cv2.copyMakeBorder(imgL, top=0, bottom=0, left=np.uint16(minDisparities/rez), right=0, borderType= cv2.BORDER_CONSTANT, value=[155,155,155] )
        filteredImg = wls_filter.filter(displ, imgLb, None, dispr)
        filteredImg = filteredImg * rez
        filteredImg = filteredImg + (cv2.getTrackbarPos('Bright','settings')-100)
        filteredImg = (cv2.getTrackbarPos('Contrast','settings')/10.0)*(filteredImg - 128) + 128
        filteredImg = np.clip(filteredImg, 0, 255)
        filteredImg = np.uint8(filteredImg)
        filteredImg = cv2.resize(filteredImg,(width,int(height/2)), interpolation = cv2.INTER_CUBIC)     # Disparity truncation hack
        filteredImg = filteredImg[0:height, np.uint16(minDisparities/rez):width]                    #
        filteredImg = cv2.resize(filteredImg,(width,int(height/2)), interpolation = cv2.INTER_CUBIC)     # Disparity truncation hack

        cv2.imshow('Depth Map', filteredImg)
        cv2.resizeWindow('Depth Map', 1000,500)
        if (value > 0):
            return filteredImg
    else:
        print ('Resolution must be greater than 0.')

def SaveDepth(value):
    if (shouldirun == 1 & value == 1):
        if (numberofimages > 1):
            if not os.path.exists(pathname):
                os.makedirs(pathname)
            print ('Saving to: ' + pathname + '\\')
            index = 1
            while (index <= numberofimages):
                filenamez = os.path.splitext(os.path.basename(sys.argv[index]))[0]
                cv2.imwrite(pathname + '\\' + filenamez + '_depthmap.jpg', Update(index), [cv2.IMWRITE_JPEG_QUALITY, 100])
                print ('%0.2f' %(100*index/numberofimages) + '%')
                index = index + 1
            print ('Batch save complete.')
        else:
            cv2.imwrite(pathname + '\\' + filename + '_depthmap.jpg', Update(1), [cv2.IMWRITE_JPEG_QUALITY, 100])            
            print ('Saved: ' + pathname + '\\' + filename + '_depthmap.jpg')
    cv2.setTrackbarPos('Save Depth','settings',0)
    
def Save6DoF(value):
    if (shouldirun == 1 & value == 1):
        if (numberofimages > 1):
            if not os.path.exists(pathname):
                os.makedirs(pathname)
            print ('Saving to: ' + pathname + '\\')
            index = 1
            while (index <= numberofimages):
                filenamez = os.path.splitext(os.path.basename(sys.argv[index]))[0]
                imgz = cv2.imread(sys.argv[index])
                heightz, widthz = imgz.shape[:2]
                #imgLz = imgz[0:int((height/2)), 0:width]
                imgRz = imgz[int((heightz/2)):heightz, 0:widthz]
                depthz = Update(index)
                depthz = cv2.cvtColor(depthz, cv2.COLOR_GRAY2RGB)
                dofz = np.concatenate((imgRz, depthz), axis=0)
                cv2.imwrite(pathname + '\\' + filenamez + '_6DoF.jpg', dofz, [cv2.IMWRITE_JPEG_QUALITY, 100])
                print ('%0.2f' %(100*index/numberofimages) + '%')
                index = index + 1
            print ('Batch save complete.')
        else:
            depth = Update(1)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
            dof = np.concatenate((imggR, depth), axis=0)
            cv2.imwrite(pathname + '\\' + filename + '_6DoF.jpg', dof, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print ('Saved: ' + pathname + '\\' + filename + '_6DoF.jpg')
    cv2.setTrackbarPos('Save 6DoF','settings',0)

shouldirun = 0
cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
cv2.namedWindow('settings', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('settings', 500,400)
cv2.namedWindow('Left', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Left', 300,150)
cv2.namedWindow('Right', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Right', 300,150)
cv2.imshow('Left', imggL)
cv2.imshow('Right', imggR)
cv2.createTrackbar('resolution','settings',0,20,Nothing)
cv2.setTrackbarPos('resolution','settings',4)
cv2.createTrackbar('blockSize','settings',0,25,Nothing)
cv2.setTrackbarPos('blockSize','settings',5)
cv2.createTrackbar('windowSize','settings',0,15,Nothing)
cv2.setTrackbarPos('windowSize','settings',5)
cv2.createTrackbar('FilterCap','settings',0,100,Nothing)
cv2.setTrackbarPos('FilterCap','settings',63)
cv2.createTrackbar('lmbda','settings',0,100,Nothing)
cv2.setTrackbarPos('lmbda','settings',80)
cv2.createTrackbar('Bright','settings',0,200,Nothing)
cv2.setTrackbarPos('Bright','settings',100)
cv2.createTrackbar('Contrast','settings',0,30,Nothing)
cv2.setTrackbarPos('Contrast','settings',10)
cv2.createTrackbar('Save Depth','settings',0,1,SaveDepth)
cv2.createTrackbar('Save 6DoF','settings',0,1,Save6DoF)
shouldirun = 1
Update(0)

while(1):
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
