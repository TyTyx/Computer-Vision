""" Example of using OpenCV API to detect and draw checkerboard pattern"""
import numpy as np
import cv2

# These two imports are for the signal handler
import signal
import sys

from itertools import product

#### Some helper functions #####
def reallyDestroyWindow(windowName) :
    ''' Bug in OpenCV's destroyWindow method, so... '''
    ''' This fix is from http://stackoverflow.com/questions/6116564/ '''
    cv2.destroyWindow(windowName)
    for i in range (1,5):
        cv2.waitKey(1)

def shutdown():
        ''' Call to shutdown camera and windows '''
        global cap
        cap.release()
        reallyDestroyWindow('img')

def signal_handler(signal, frame):
        ''' Signal handler for handling ctrl-c '''
        shutdown()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
##########

############## calibration of plane to plane 3x3 projection matrix

def compute_homography(fp,tp):
    ''' Compute homography that takes fp to tp.
    fp and tp should be (N,3) '''

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # create matrix for linear method, 2 rows for each correspondence pair
    num_corners = fp.shape[0]

    # construct constraint matrix
    A = np.zeros((num_corners*2,9));
    A[0::2,0:3] = fp
    A[1::2,3:6] = fp
    A[0::2,6:9] = fp * -np.repeat(np.expand_dims(tp[:,0],axis=1),3,axis=1)
    A[1::2,6:9] = fp * -np.repeat(np.expand_dims(tp[:,1],axis=1),3,axis=1)

    # solve using *naive* eigenvalue approach
    D,V = np.linalg.eig(A.transpose().dot(A))

    H = V[:,np.argmin(D)].reshape((3,3))

    # normalise and return
    return H

##############


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# YOU SHOULD SET THESE VALUES TO REFLECT THE SETUP
# OF YOUR CHECKERBOARD
WIDTH = 6
HEIGHT = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:HEIGHT,0:WIDTH].T.reshape(-1,2)

cap = cv2.VideoCapture(0)

## Step 0: Load the image you wish to overlay
im = cv2.imread('alone.jpg')

cutx = np.linspace(640,0,9)
cuty = np.linspace(0,480,6)
tp = np.c_[np.asarray(list(product(cutx, cuty))),np.ones(54)]

while (True):
        #capture a frame
        ret, img = cap.read()

        ## IF YOU WISH TO UNDISTORT YOUR IMAGE YOU SHOULD DO IT HERE

        # Our operations on the frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (HEIGHT,WIDTH),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            cv2.drawChessboardCorners(img, (HEIGHT, WIDTH), corners, ret)
            # HEIGHT = 9
            # WIDTH = 6

            ## Step 1a: Compute fp -- an Nx3 array of the 2D homogeneous coordinates of the
            ## detected checkerboard corners

            corners = np.asmatrix(corners)
            fp = np.asarray(np.c_[corners, np.ones(corners.shape[0])])

            ## Step 1b: Compute tp -- an Nx3 array of the 2D homogeneous coordinates of the
            ## samples of the image coordinates
            ## Note: this could be done outside of the loop either!

            tp = np.asarray(tp)

            ## Step 2: Compute the homography from tp to fp

            H = compute_homography(tp, fp)

            ## Step 3: Compute warped mask image

            tp = np.asmatrix(tp)
            im[:,2] = 0
            H = np.asarray(H)
            wpimg = cv2.warpPerspective(im, H, dsize=(640, 480))

            ## Step 4: Compute warped overlay image

            rows,cols,chans = wpimg.shape
            roi = img[0:rows,0:cols]
            #convert it to gray scale
            graysrc = cv2.cvtColor(wpimg,cv2.COLOR_BGR2GRAY)
            #creating the mask, and the it's inverse
            ret,mask = cv2.threshold(graysrc, 10, 255 , cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            #black out the area in ROI, this is where the imposed image would go
            img1_bg  = cv2.bitwise_and(roi,roi,mask = mask_inv)
            #i take only the region of our 'src' image
            img2_fg = cv2.bitwise_and(wpimg,wpimg,mask=mask)
            #puts the two together and the modify the main image
            dst = cv2.add(img1_bg,img2_fg)
            img[0:rows,0:cols] = dst

            ## Step 5: Compute final image by combining the warped frame with the captured frame

        cv2.imshow('img',img)
        cv2.imshow('img', wpimg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release everything
shutdown()
