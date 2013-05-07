import cv2 as cv
import timeit
import time

NUMBER_OF_TESTS = 100

#cv.namedWindow("Camera Capture")
capture = cv.VideoCapture(-1)



fastest = 1000000.0
slowest = 0.0
start = 0.0
end = 0.0
total = 0.0
average = 0.0

def repeat():
#    global fastest, slowest
#    start = time.time()
    rval, frame = capture.read()
    key = cv.waitKey(20)
#    total = time.time() - start
#    if fastest > total:
#        fastest = total
#    elif slowest < total:
#        slowest = total
#    if key == 27:
#        break
#    frame = cv.QueryFrame(capture)
#    cv.ShowImage("Camera Capture", frame)

if __name__ == "__main__":
    
    if capture.isOpened():
        print "Capture is open."
        rval, frame = capture.read()
    else:
        print "Capture is not open."
        rval = False
    
    for i in range(NUMBER_OF_TESTS):
#        global fastest, slowest
        start = time.time()
        repeat()
        total = time.time()-start
        average = total / NUMBER_OF_TESTS
        print i, " - ", total
    
    print average
#        if i % 10 == 0:
#            print i
        
#    print "Fastest: ", fastest
#    print "Slowest: ", slowest
#        print(timeit.timeit("repeat()", setup="from __main__ import repeat"))
    
#    while rval:
#        cv.imshow("Camera Capture", frame)
#        rval, frame = capture.read()
#        key = cv.waitKey(20)
#        if key == 27: # exit on ESC
#            break
#        repeat()
    
#    print "Loading image..."
#    im = cv.LoadImageM("nathan.jpg")
#    
#    print "Image type: "
#    print type(im)
#    
#    print "Finding good features to track..."
#    img = cv.LoadImageM("nathan.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
#    eig_image = cv.CreateMat(img.rows, img.cols, cv.CV_32FC1)
#    temp_image = cv.CreateMat(img.rows, img.cols, cv.CV_32FC1)
#    for (x,y) in cv.GoodFeaturesToTrack(img, eig_image, temp_image, 10, 0.04, 1.0, useHarris = True):
#        print "good feature at ", x, y
#        
#    cv.NamedWindow("Source", 1)
#    cv.ShowImage("Source", im)
#    
#    dest = im
#    cv.Smooth(im, dest)
#    
#    cv.NamedWindow("Destination", 1)
#    cv.ShowImage("Destination", dest)
#    
#    cv.WaitKey(0)
