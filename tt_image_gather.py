import cv2 as cv
import time

NUMBER_OF_TESTS = 100

capture = cv.VideoCapture(-1)

start = 0.0
total = 0.0
average = 0.0

def repeat():
    rval, frame = capture.read()
    key = cv.waitKey(20)

if __name__ == "__main__":
    
    if capture.isOpened():
        print "Capture is open."
        rval, frame = capture.read()
    else:
        print "Capture is not open."
        rval = False
    
    for i in range(NUMBER_OF_TESTS):
        start = time.time()
        repeat()
        total = time.time()-start
        average = average + total / NUMBER_OF_TESTS
        print i, "-", total
    
    print average
