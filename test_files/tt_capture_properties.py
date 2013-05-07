#!/usr/bin/env python

import cv2
import time

NUMBER_OF_TESTS = 1000

capture = cv2.VideoCapture(-1)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 100)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 100)

start = 0.0
total = 0.0
average = 0.0

def repeat():
    rval, frame = capture.read()
    key = cv2.waitKey(1)
    frame = cv2.flip(frame,1)
    cv2.imshow('Frame', frame)
    # print (frame.__dict__)
    # width, height = frame.width
    # print width, height
    

if __name__ == "__main__":
    
    if capture.isOpened():
        print "Capture is open."
        rval, frame = capture.read()
    else:
        print "Capture is not open."
        rval = False
    
    # while True:
        # repeat()
    for i in range(NUMBER_OF_TESTS):
        start = time.time()
        repeat()
        total = time.time()-start
        average = average + total
        print i, "-", total, "---", capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    
    print capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    print capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    
    print average / NUMBER_OF_TESTS
