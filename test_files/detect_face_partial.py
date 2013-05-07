import cv2 as cv
import sys



def detect(image, cascade):
	
	# Image preprocessing
	img_copy = cv.resize(image, (image.shape[1]/2, image.shape[0]/2))
	gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
	gray = cv.equalizeHist(gray)
	
	# Detect the faces
	rects = cascade.detectMultiScale(gray)
	
	# Make a copy, as we don't want to draw on the original image
	for x, y, width, height in rects:
		cv.rectangle(img_copy, (x, y), (x+width, y+height), (255, 0, 0), 2)
	cv.imshow('Camera', img_copy)
	
	# image_size = cv.GetSize(image)
	
	# create grayscale version
	# grayscale = cv.CreateImage(image_size, 8, 1) #(size, depth, channels)
	# cv.CvtColor(image, grayscale, cvBGR2GRAY) #(source, destination, conversion code)
	
	# create storage
	#storage = cv.CreateMemStorage(0)
	#cv.ClearMemStorage(storage)
	
	# equalize histogram
	# cv.EqualizeHist(grayscale, grayscale)
	
	# detect objects
	#cascade = cv.LoadHaarClassifiedCascade('../data/haarcascades/haarcascade_frontalface_alt.xml', cv.Size(1,1))
	#faces = cv.HaarDetectObjects(image, cascade, storage, 1.2, 2, cv.HAAR_DO_CANNY_PRUNING, cv.Size(50, 50))
	
	#if faces:
	#	print 'face detected!'
	#	for i in faces:
	#		cv.Rectangle(image, cv.Point( int(i.x), int(i.y)), cv.Point( int(i.x +i.width), int(i.y + i.height)), cv.RGB(0, 255, 0), 3, 8, 0)

if __name__ == "__main__":
	# create window
	cv.namedWindow('Camera', 1)
	
	# set camera source and create capture device
	video_src = -1 # select first (and only) camera
	capture = cv.VideoCapture(-1)

	# select cascade file and create a new CascadeClassifier from given file
	cascade_fn = "../data/haarcascades/haarcascade_frontalface_default.xml"
	cascade = cv.CascadeClassifier(cascade_fn)
	
	# check if capture device is OK
	if not capture.isOpened():
		print "Error opening capture device"
		sys.exit(1)
	
	# loop forever
	while 1:
		
		# capture the current frame
		rval, frame = capture.read()
		if not rval:
			break
		
		# mirror
		frame = cv.flip(frame, 1)
		
		# face detection
		detect(frame, cascade)
		
		# display webcam image
		# cv.imshow('Camera', frame)
		
		# handle events
		k = cv.waitKey(10)
		
		if k == 0x1b: # ESC
			print 'ESC pressed. Exiting ...'
			break
