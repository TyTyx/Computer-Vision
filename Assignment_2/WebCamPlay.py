""" Simple example code for reading an image from a webcam
and manipulating pixels within the resulting image """

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while (True):
        #capture a frame
        ret, img = cap.read()

       # OUr operations on the frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	    # gray[1:200:2,1:200:2] = 0

        cv2.imshow('img',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release everything
cap.release()
cv2.destroyAllWindows()
