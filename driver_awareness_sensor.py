import numpy as np
import cv2
import cv2.cv as cv
import time
import numpy as np
import sys

####################################################################################################
# CONFIGURATION
#################################################################################################### 

# location of face cascade file
FACE_CASCADE_FN = "./data/haarcascades/haarcascade_frontalface_default.xml"

# location of eye cascade file
EYE_CASCADE_FN = "./data/haarcascades/haarcascade_eye.xml"

# camera capture width
# default is 640
# possible values seem to be 160, 176, 320, 352, 640
CAPTURE_WIDTH = 320

NUMBER_OF_TESTS = None   # sets the number of frames to grab and process, set to None to disable

### visual output settings

# if SHOW is false, all SHOW_ settings are false
# if SHOW is true, SHOW_ settings keep their assigned values
SHOW = True

SHOW_DEBUG = True           # shows print statements in the console

SHOW_GRAYSCALE = True       # shows the grayscale image used by the detectMultiScale function
SHOW_RT_FACE = True         # shows all detected faces for each frame
SHOW_SMOOTH_FACE = True     # shows the smoothed_face rect
SHOW_RT_EYES = True         # shows all detected eyes for each frame
SHOW_SMOOTH_EYES = True     # shows the smoothed_eyes rects
SHOW_STATS = True           # shows the stats overlaid on the image

TIMEIT = True               # records time after each part, may slow down process

####################################################################################################
# End Configuration
####################################################################################################

CAPTURE_HEIGHT = CAPTURE_WIDTH * 3 / 4

X1, Y1, X2, Y2 = 0, 1, 2, 3

if not SHOW:
    SHOW_GRAYSCALE = False
    SHOW_RT_FACE = False
    SHOW_SMOOTH_FACE = False
    SHOW_RT_EYES = False
    SHOW_SMOOTH_EYES = False
    SHOW_STATS = False

#---------------------------------------------------------------------------------------------------
# read_frame() returns the next frame from the camera.
# Receive: camera, the camera to use for capture
# Return: frame, the next frame from camera
#---------------------------------------------------------------------------------------------------
def read_frame(camera):
    frame_found, frame = cam.read()
    while not frame_found:
        frame_found, frame = cam.read()
    return frame

#---------------------------------------------------------------------------------------------------
# preprocess() processes the image to make it smaller to increase future processing time.
# Receive: image, the image to be processed
# Return: processed_image, the imaged that has been processed
#---------------------------------------------------------------------------------------------------
def preprocess(image):
    # reduce the number of channels
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # compensate for lighting variations
    processed_image = cv2.equalizeHist(processed_image)
    
    return processed_image
    
#---------------------------------------------------------------------------------------------------
# detect_faces() uses the given cascade to find the desired features.
# Receive: img, an image
#          cascade, the cascade to use to locate the features
# Return: rects, a list of rectangles, each represented as numpy.ndarray
#---------------------------------------------------------------------------------------------------
def detect_faces(img, cascade):
    #-----------------------------------------------------------------------------------------------
    # detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]])
    # image - Matrix of the type CV_8U containing an image where objects are detected.
    # scaleFactor - Parameter specifying how mucht the image size is reduced at each image object.
    # minNeighbors - Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # flags - Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
    # minSize - Minimum possible object size. Objects smaller than that are ignored.
    # maxSize - Maximum possible object size. Objects larger than that are ignored.
    #-----------------------------------------------------------------------------------------------
    height = len(img)
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=(height/3, height/3), flags = cv.CV_HAAR_SCALE_IMAGE)
    
    if len(rects) == 0:
        return []
    
    rects[:,2:] += rects[:,:2] # transforms [x1, y1, width, length] into [x1, y1, x2, y2]
    return rects

def detect_eyes(img, cascade):
    # height = len(img)
    min_eye_size = len(img)/5
    #TODO? max_eye_size = len(img)/2
    rects = cascade.detectMultiScale(img, scaleFactor=2, minNeighbors=4, minSize=(min_eye_size, min_eye_size), flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

#---------------------------------------------------------------------------------------------------
# choose_face() selects the desired face from possible faces. The desired face is determined based
# off proximity to the previous face. If no faces are passed, the method will return the same empty 
# list. If one face is passed, the proximity will be checked. If it is not within the acceptable
# range, the proximity counter will be increased, in hopes of finding the face next time in a larger
# range. If there are multiple faces, the method will look for the closest face to the previously
# selected face. It will then check if it is within the acceptable proximity and, if so, return it.
# If not, the proximity counter will be increased.
# Receive: face_rects, a list of numpy arrays containing x1, y1, x2, y2 for each face detected
# Return: selected_face, a numpy array containing x1, y1, x2, y2
#---------------------------------------------------------------------------------------------------
previous_face_rect = np.array([CAPTURE_WIDTH/2-5, CAPTURE_HEIGHT/2-5, CAPTURE_WIDTH/2+5, CAPTURE_HEIGHT/2+5])
previous_proximity = np.zeros(4, dtype=np.uint)
P_CHANGE = CAPTURE_WIDTH/160

def choose_face(face_rects):
    global previous_proximity, previous_face_rect
    
    print "previous_proximity: ", previous_proximity
    
    if len(face_rects) == 0:
        previous_proximity += P_CHANGE
        return None
    
    elif len(face_rects) == 1:
        face_rect_array = face_rects[0]
        confidence = 0
        
        if abs(face_rect_array[X1] - previous_face_rect[X1]) < previous_proximity[X1]:
            confidence += 1
            previous_proximity[X1] -= P_CHANGE if previous_proximity[X1] >= P_CHANGE else 0
        else:
            confidence -= 1
            previous_proximity[X1] += P_CHANGE if previous_proximity[X1] <= CAPTURE_WIDTH else 0
            
        if abs(face_rect_array[Y1] - previous_face_rect[Y1]) < previous_proximity[Y1]:
            confidence += 1
            previous_proximity[Y1] -= P_CHANGE if previous_proximity[Y1] >= P_CHANGE else 0
        else:
            confidence -= 1
            previous_proximity[Y1] += P_CHANGE
        
        if abs(face_rect_array[X2] - previous_face_rect[X2]) < previous_proximity[X2]:
            confidence += 1
            previous_proximity[X2] -= P_CHANGE if previous_proximity[X2] >= P_CHANGE else 0
        else:
            confidence -= 1
            previous_proximity[X2] += P_CHANGE
        
        if abs(face_rect_array[Y2] - previous_face_rect[Y2]) < previous_proximity[Y2]:
            confidence += 1
            previous_proximity[Y2] -= P_CHANGE if previous_proximity[Y2] >= P_CHANGE else 0
        else:
            confidence -= 1
            previous_proximity[Y2] += P_CHANGE
        
        # print "confidence: ", confidence
        
        if confidence >= 0:
            previous_face_rect = face_rect_array
            return face_rect_array
        else:
            return None
    
    else:
        confidence = np.zeros(len(face_rects))
        counter = 0
        
        # generate confidence for all face_rects
        for x1, y1, x2, y2 in face_rects:
            confidence[counter] += 1 if abs(x1 - previous_face_rect[X1]) < previous_proximity[X1] else -1
            confidence[counter] += 1 if abs(y1 - previous_face_rect[Y1]) < previous_proximity[Y1] else -1
            confidence[counter] += 1 if abs(x2 - previous_face_rect[X2]) < previous_proximity[X2] else -1
            confidence[counter] += 1 if abs(y2 - previous_face_rect[Y2]) < previous_proximity[Y2] else -1
            
        best = -1
        best_conf = -4
        
        # find best face_rect
        for i in range(len(face_rects)):
            if confidence[i] >= best_conf:
                best = i
                best_conf = confidence[i]
        
        # update proximities and return best face_rect
        if best_conf >= 0:
            face_rect_array = face_rects[best]
            
            # add P_CHANGE if point is too far away, else subtract P_CHANGE if value will not result in zero
            previous_proximity[X1] += P_CHANGE if abs(face_rect_array[X1] - previous_face_rect[X1]) >= previous_proximity[X1] and previous_proximity[X1] < CAPTURE_WIDTH else (-P_CHANGE if previous_proximity[X1] >= P_CHANGE else 0)
            previous_proximity[Y1] += P_CHANGE if abs(face_rect_array[Y1] - previous_face_rect[Y1]) >= previous_proximity[Y1] and previous_proximity[Y1] < CAPTURE_HEIGHT else (-P_CHANGE if previous_proximity[Y1] >= P_CHANGE else 0)
            previous_proximity[X2] += P_CHANGE if abs(face_rect_array[X2] - previous_face_rect[X2]) >= previous_proximity[X2] and previous_proximity[X2] < CAPTURE_WIDTH else (-P_CHANGE if previous_proximity[X2] >= P_CHANGE else 0)
            previous_proximity[Y2] += P_CHANGE if abs(face_rect_array[Y2] - previous_face_rect[Y2]) >= previous_proximity[Y2] and previous_proximity[Y2] < CAPTURE_HEIGHT else (-P_CHANGE if previous_proximity[Y2] >= P_CHANGE else 0)
            
            previous_face_rect = face_rects[best]
            
            return face_rect_array
        
        else:
            return None
        
    selected_face = face_rects[0]
    
    return selected_face
    
#---------------------------------------------------------------------------------------------------
# smooth_face() smooths the face rectangle by averaging a set number of previous face rectangles.
# Receive: face_rect, a numpy array containing x1, y1, x2, y2
# Return: smoothed_face, a numpy array containing x1, y1, x2, y2
#---------------------------------------------------------------------------------------------------
SMOOTH_FACE_MAX = 5 # number of frames to take face rect average
smooth_face_history = np.zeros((SMOOTH_FACE_MAX, 4), dtype=np.int32)
smooth_face_history[0] = [0, 0, SMOOTH_FACE_MAX, SMOOTH_FACE_MAX] # avoids initial size of zero for smooth_face() if no face is found
smooth_face_counter = 0

def smooth_face(face_rect):
    global smooth_face_counter, smooth_face_history
    
    # if face_rect contains a face, add it to the history, replacing the oldest
    if face_rect != None:
        smooth_face_history[smooth_face_counter] = face_rect
        smooth_face_counter = (smooth_face_counter + 1) % SMOOTH_FACE_MAX
    
    smoothed_face = np.mean(smooth_face_history, axis=0, dtype=np.int)
    print "smooth_face_mean: ", smoothed_face
    
    # x1sum = y1sum = x2sum = y2sum = 0
    
    # for x1, y1, x2, y2 in smooth_face_history:
        # x1sum += x1
        # y1sum += y1
        # x2sum += x2
        # y2sum += y2
    
    # smoothed_face = np.array([int(x1sum/SMOOTH_FACE_MAX), int(y1sum/SMOOTH_FACE_MAX), int(x2sum/SMOOTH_FACE_MAX), int(y2sum/SMOOTH_FACE_MAX)])
    
    # smoothed_face = smooth_face_mean / SMOOTH_FACE_MAX
    # print "smoothed_face: ", smoothed_face
    
    return smoothed_face
    
#---------------------------------------------------------------------------------------------------
# choose_eyes() attempts to select the desired eyes from the possible detected eyes. The method will
# use the proximity of previously located eyes and the relative location within the face in order to
# keep the eyes separate from each other. This is important for the detect_eye_state() function so
# it can compare the same eye each time.
# Receive: eye_rects, a list of numpy arrays containing x1, y1, x2, y2 for each eye detected
# Return: selected_eyes, a list of numpy arrays containing x1, y1, x2, y2 for both eyes
#---------------------------------------------------------------------------------------------------
previous_eye_rect = np.zeros(4)

def choose_eyes(eye_rects):
    #TODO: implement method
    selected_eyes = []
    return selected_eyes

#---------------------------------------------------------------------------------------------------
# smooth_eyes() smooths the eye rectangles by averaging a set number of previous eye rectangles, for
# both eyes.
# Receive: eye_rects, a list of numpy arrays containing x1, y1, x2, y2 for both eyes
# Return: smoothed_eyes, a list of numpy arrays containing x1, y1, x2, y2 for both eyes
#---------------------------------------------------------------------------------------------------
def smooth_eyes(eye_rects):
    #TODO: implement method
    smoothed_eyes = []
    return smoothed_eyes
    
#---------------------------------------------------------------------------------------------------
# detect_eye_state() determines the state of each eye, whether open or closed (or in between?).
# Receive: eye_rects, a list of numpy arrays containing x1, y1, x2, y2 for both eyes
# Return: eye_state, a yet to be determined value of a yet to be determined type
#---------------------------------------------------------------------------------------------------
def detect_eye_state(eye_rects):
    #TODO: implement method
    eye_state = []
    return eye_state
    
####################################################################################################
# Drawing functions for demo and debug.
####################################################################################################

#---------------------------------------------------------------------------------------------------
# draw_rects() draws a rectangle on the given image using the coordinates provided and line width of
# two pixels.
# Receive: img, an image
#          rects, a list of rectangles, each represented as numpy.ndarray
#          color, the color of the rectangle
#---------------------------------------------------------------------------------------------------
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def draw_rect(img, rect, color):
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
    
#---------------------------------------------------------------------------------------------------
# draw_text() draws the given text on the image at the specified coordinates.
# Receive: img, an image
#          text, the text to draw on the image
#          (x, y), the coordinates of the lower left corner
#---------------------------------------------------------------------------------------------------
def draw_text(img, text, (x, y)):
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2, cv2.CV_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1, cv2.CV_AA)
    
####################################################################################################
# The main code, where it all goes down...
####################################################################################################
if __name__ == '__main__':
    
    print "Starting Driver Awareness Sensor..."
    print "Running (ESC in window to exit)..."
        
    # create cascades for face and eyes
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FN)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_FN)
    
    # create video capture
    cam = cv2.VideoCapture(-1)
    if not cam.isOpened():
        print "No camera found. Exiting..."
        sys.exit()
    
    # set capture settings
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    
    runtime_sum = 0.0000001 # prevent divide by zero
    counter = 1
    
    frame_read_time, face_detect_time, frames_per_second = 0, 0, 0
    display_frt, display_fdt, display_fps = 0, 0, 0
    # counter = 0
    previous_display_time = 0
    
    color = 0
    
    
    
    average = 0.0
    i = 0
    
    while True:
        if SHOW_DEBUG:
            print "------------------------------------------"
        ##### read image from camera
        if TIMEIT:
            t1 = time.time()
        
        frame = read_frame(cam)
        img = frame #REMOVE
        
        
        ##### convert to grayscale and equalize
        if TIMEIT:
            t2 = time.time()
            
        gray = preprocess(frame)
        
        
        ##### detect faces in image
        if TIMEIT:
            t3 = time.time()
        
        # print len(gray), len(gray[0])
        face_rects = detect_faces(gray, face_cascade)
        faces_found = len(face_rects) # number of faces found
        
        
        ##### choose correct face from detected faces
        if TIMEIT:
            t4 = time.time()
        
        face_rect = choose_face(face_rects)
        if SHOW_DEBUG:
            print "face_rect: ", face_rect
        
        
        ##### smooth face using recent average
        if TIMEIT:
            t5 = time.time()
        
        smoothed_face = smooth_face(face_rect)
        if SHOW_DEBUG:
            print "smoothed_face: ", smoothed_face
        
        
        ##### detect eyes in smoothed face
        if TIMEIT:
            t6 = time.time()
        
        face_area = gray[smoothed_face[Y1]:smoothed_face[Y2], smoothed_face[X1]:smoothed_face[X2]]
        eye_rects = detect_eyes(face_area.copy(), eye_cascade)
        eyes_found = len(eye_rects)
        
        
        ##### choose correct eyes from detected eyes
        if TIMEIT:
            t7 = time.time()
        
        
        
        
        ##### smooth eyes using recent average
        if TIMEIT: 
            t8 = time.time()
        
        
        ##### detect the state of the eyes
        if TIMEIT:
            t9 = time.time()
        
        
        ##### output alert if state of eyes is dangerous
        if TIMEIT:
            t10 = time.time()
        
        
        ##### Display stuff for debug/demo
        if TIMEIT:
            t11 = time.time()
        
        if SHOW:
            display_image = img.copy()
            
            # Draw all faces
            if SHOW_RT_FACE:
                draw_rects(display_image, face_rects, (255, 255, 255))
            
            # Draw smoothed face
            if SHOW_SMOOTH_FACE:
                if not face_rect == None:
                    color = color + 16 if color <= 239 else 255
                else:
                    color = color - 16 if color >= 15 else 0
                
                draw_rect(display_image, smoothed_face, (0, color, (255 - color)))
            
            # Draw all eyes
            if SHOW_RT_EYES:
                display_image_face_area = display_image[smoothed_face[Y1]:smoothed_face[Y2], smoothed_face[X1]:smoothed_face[X2]]
                draw_rects(display_image_face_area, eye_rects, (255, 0, 0))
            
            display_image = cv2.flip(display_image, 1)
            
            # Print stats on image
            if SHOW_STATS and TIMEIT:
                draw_text(display_image, 'Average time: %.1f ms' % (runtime_sum/counter*1000), (20, 20))
                draw_text(display_image, 'Average fps: %.1f' % (1/(runtime_sum/counter)), (20, 40))
                            
            cv2.imshow('Face Detector', display_image)
            
            if SHOW_GRAYSCALE:
                gray = cv2.flip(gray, 1)
                cv2.imshow('Grayscale', gray)
            
        # Display stats on console
        runtime_sum += t11 - t1
        
        if SHOW_DEBUG:
            print counter, " - Faces: ", faces_found, " -- Eyes: ", eyes_found
        
        #####
        
        if not NUMBER_OF_TESTS == None:
            sys.stdout.write("\rCompleted test %i" % counter)
            sys.stdout.flush()
        
        counter = counter + 1
        
        # Exit on ESC
        # The waitKey function is very necessary.
        # Besides waiting the specified seconds for a key press (and returning -1 if none),
        # it also handles any windowing events, such as creating windows with imshow()
        # http://stackoverflow.com/questions/5217519/opencv-cvwaitkey
        # TODO: Wait for key only if showing images
        if cv2.waitKey(5) == 27:
            break
        
        # Exit after predefined number of tests, if set
        if not NUMBER_OF_TESTS == None:
            if counter > NUMBER_OF_TESTS:
                sys.stdout.write("\n\n")
                print NUMBER_OF_TESTS, "tests completed. Exiting..."
                break
    
    print "Stopped."
    print "Average loop time (ms):", (int(runtime_sum/counter*10000))/10.0, "--> fps:", (int(1/(runtime_sum/counter)*10)/10.0)
            
