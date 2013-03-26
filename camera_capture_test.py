import cv

if __name__ == "__main__":
    
    print "Loading image..."
    im = cv.LoadImageM("nathan.jpg")
    
    print "Image type: "
    print type(im)
    
    print "Finding good features to track..."
    img = cv.LoadImageM("nathan.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
    eig_image = cv.CreateMat(img.rows, img.cols, cv.CV_32FC1)
    temp_image = cv.CreateMat(img.rows, img.cols, cv.CV_32FC1)
    for (x,y) in cv.GoodFeaturesToTrack(img, eig_image, temp_image, 10, 0.04, 1.0, useHarris = True):
        print "good feature at ", x, y
        
    cv.NamedWindow("Source", 1)
    cv.ShowImage("Source", im)
    
    dest = im
    cv.Smooth(im, dest)
    
    cv.NamedWindow("Destination", 1)
    cv.ShowImage("Destination", dest)
    
    cv.WaitKey(0)
