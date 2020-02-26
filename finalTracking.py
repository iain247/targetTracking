# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:58:41 2019

@author: iainr
"""

#import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

#define midpoint function
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#allow access to webcam
image = VideoStream(src=0).start()

#define line width and colours for later use
width = 2.5 
red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)

while True:
	#grab the current frame and initialize the status text
    frame = image.read()
    status = "No Targets"
    
    #define centre of video
    Ox = frame.shape[1] / 2
    Oy = frame.shape[0] / 2
    
	#convert the frame to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)
 
    #find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    #loop over the contours
    for c in cnts:
		#approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
 
		#ensure that the approximated contour is "roughly" rectangular
        if len(approx) >= 4 and len(approx) <= 6:
			#compute the bounding box of the approximated contour and
			#use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRatio = w / float(h)
 
			#compute the solidity of the original contour
            area = cv2.contourArea(c)
            hullArea = cv2.contourArea(cv2.convexHull(c))
            solidity = area / float(hullArea)
 
			#compute whether or not the width and height, solidity, and
			#aspect ratio of the contour falls within appropriate bounds
            keepDims = w > 25 and h > 25
            keepSolidity = solidity > 0.9
            keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2
 
			#ensure that the contour passes all our tests
            if keepDims and keepSolidity and keepAspectRatio:
				#draw an outline around the target and update the status
				#text
                cv2.drawContours(frame, [approx], -1, green, 4)
                status = "Target Acquired"
 
                #compute the rotated bounding box of the contour
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

	            #compute the center of the bounding box
                cX = np.average(box[:, 0])
                cY = np.average(box[:, 1])
              
		        #unpack the ordered bounding box, then compute the
                #midpoint between the top-left and top-right points,
                #followed by the midpoint between the top-right and
                #bottom-right
                (tl, tr, br, bl) = box
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                #compute the Euclidean distance between the midpoints,
                #then construct the reference object
                refD = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                refObj = ((Ox, Oy), refD / width)
    
	            #define referance coordinates and object coordinates
                refCoords = refObj[0]
                objCoords = (cX, cY)
    
	            #define x and y coordinates
                xA = refCoords[0]
                yA = refCoords[1]
                xB = objCoords[0]
                yB = objCoords[1]
    
	            #draw circles corresponding to the current points and
	            #connect them with a line
                cv2.circle(frame, (int(xA), int(yA)), 5, green, -1)
                cv2.circle(frame, (int(xB), int(yB)), 5, green, -1)
                cv2.line(frame, (int(xA), int(yA)), (int(xB), int(yB)),
                    green, 2)

	            #compute the Euclidean distance between the coordinates,
	            #and then convert the distance in pixels to distance in
	            #units
                D = dist.euclidean((xA, yA), (xB, yB)) / refObj[1]
                (mX, mY) = midpoint((xA, yA), (xB, yB))
                cv2.putText(frame, "{:.1f} cm".format(D), (int(mX), int(mY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 2)
                
                #draw the status text on the frame
                cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, red, 2)
                
                #draw the x and y distances on the frame
                Dx = (xA-xB)/refObj[1]
                Dy = (yA-yB)/refObj[1]
                cv2.putText(frame, "Dx = {:.1f} cm".format(Dx), (20, 455), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                cv2.putText(frame, "Dy = {:.1f} cm".format(Dy), (200, 455), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
 
    #show the frame and record if a key is pressed
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
	#if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
 
#cleanup the camera and close any open windows
cv2.destroyAllWindows()