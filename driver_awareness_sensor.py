import numpy as np
import cv2
import cv2.cv as cv
import time
import numpy as np

face_cascade_fn = "./data/haarcascades/haarcascade_frontalface_alt.xml"
eye_cascade_fn = "./data/haarcascades/haarcascade_eye.xml"

SMOOTH_FACE_MAX = 10
smooth_face_history = np.zeros((SMOOTH_FACE_MAX, 4), dtype=np.int32)
smooth_face_history[0] = [0, 0, 10, 10] # avoids initial size of zero for smooth_face() if no face is found
smooth_face_counter = 0

CAPTURE_WIDTH = 160 # default is 640 (160, 176, 320, 352, 640)
CAPTURE_HEIGHT = CAPTURE_WIDTH * 3 / 4

SHOW_REAL_TIME = True
SHOW_SMOOTH = True
SHOW_EYES = True
SHOW_STATS = False

SHOW = True

TIMEIT = True	

NUMBER_OF_TESTS = 1000

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
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(min_eye_size, min_eye_size), flags=cv.CV_HAAR_SCALE_IMAGE)
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
def choose_face(face_rects):
    #TODO: implement method
    # selected_face = np.zeros(4, dtype=np.int32)
    if len(face_rects) == 0:
        return None
    selected_face = face_rects[0]
    
    return selected_face
    
#---------------------------------------------------------------------------------------------------
# smooth_face() smooths the face rectangle by averaging a set number of previous face rectangles.
# Receive: face_rect, a numpy array containing x1, y1, x2, y2
# Return: smoothed_face, a numpy array containing x1, y1, x2, y2
#---------------------------------------------------------------------------------------------------
def smooth_face(face_rect):
    #TODO: update method to receive and return correct types
    
    global smooth_face_counter, smooth_face_history
    
    # choose which face rect to process
    # currently the first one is choosen, even if incorrect
    if len(face_rect) > 0:
        # store first rect in history array
        smooth_face_history[smooth_face_counter] = face_rect[0:1]
        smooth_face_counter += 1
        smooth_face_counter %= SMOOTH_FACE_MAX
        
        # reset counter, if necessary
        if smooth_face_counter >= SMOOTH_FACE_MAX:
            smooth_face_counter = 0
    
    x1sum = y1sum = x2sum = y2sum = 0
    
    for x1, y1, x2, y2 in smooth_face_history:
        x1sum += x1
        y1sum += y1
        x2sum += x2
        y2sum += y2
    
    smoothed_face = [np.array([int(x1sum/SMOOTH_FACE_MAX), int(y1sum/SMOOTH_FACE_MAX), int(x2sum/SMOOTH_FACE_MAX), int(y2sum/SMOOTH_FACE_MAX)])]
    
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
    
    if not SHOW:
        SHOW_REAL_TIME = False
        SHOW_SMOOTH = False
        SHOW_EYES = False
        SHOW_STATS = False
    
    # create cascades for face and eyes
    face_cascade = cv2.CascadeClassifier(face_cascade_fn)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_fn)
    
    # create video capture
    cam = cv2.VideoCapture(-1)
    
    # set capture settings
    #TODO!
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    
    frame_read_time, face_detect_time, frames_per_second = 0, 0, 0
    display_frt, display_fdt, display_fps = 0, 0, 0
    counter = 0
    previous_display_time = 0
    
    color = 0
    
    #s print "SHOW: " + str(SHOW) + " - " + str(SHOW_REAL_TIME) + " - " + str(SHOW_SMOOTH)
    
    average = 0.0
    i = 0
    while True:
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
        # print face_rect
        
        ##### smooth face using recent average
        
        ##### detect eyes in smoothed face
        
        ##### choose correct eyes from detected eyes
        
        ##### smooth eyes using recent average
        
        ##### detect the state of the eyes
        
        ##### output alert if state of eyes is dangerous
    
        ########## Begin old code ##########
        # t1 = time.time()
        
        # # read image from camera
        # ret, img = cam.read()
        
        # # create grayscale image for faster processing
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        
        # t2 = time.time()
        
        # # detect faces in image
        # face_rects = detect_faces(gray, face_cascade)
        # faces_found = len(face_rects)
        
        if SHOW:
            disp_img = img.copy()
        
        # show each detected face in real time
        if SHOW_REAL_TIME:
            draw_rects(disp_img, face_rects, (0, 255, 0))
        
        # update smooth_face rectangle
        smoothed_face = smooth_face(face_rects)
        
        eyes_found = 0
        for x1, y1, x2, y2 in smoothed_face:
        # for x1, y1, x2, y2 in rects:
            face_area = gray[y1:y2, x1:x2]
            eye_rects = detect_eyes(face_area.copy(), eye_cascade)
            eyes_found += len(eye_rects)
            
            if SHOW_EYES:
                disp_img_face_area = disp_img[y1:y2, x1:x2]
                draw_rects(disp_img_face_area, eye_rects, (255, 0, 0))
        
        t3 = time.time()
        
        if SHOW_SMOOTH:
            if len(face_rects) > 0:
                color += 16
                if color > 255:
                    color = 255
            else:
                color -= 16
                if color < 0:
                    color = 0
                    
            draw_rects(disp_img, smooth_face(face_rects), (0, color, (255-color)))
        
        t4 = time.time()
        
        average += t4-t1
        
        #----------------------------------------
        # General image processing done. The 
        # following is for demonstration and
        # debug.
        #----------------------------------------
        
        counter = counter + 1
        frame_read_time += t2 - t1
        face_detect_time += t3 - t2
        frames_per_second += t3 - t1
        
        # Update stats every second, currently does not seem to work as desired
        current_seconds = int(time.time())
        if current_seconds != previous_display_time: # display average each second
            previous_time_display = current_seconds
            display_frt = 1000*(frame_read_time / counter)
            display_fdt = 1000*(face_detect_time / counter)
            display_fps = 1/(frames_per_second / counter)
            #counter = 0
            frame_read_time, face_detect_time, frames_per_second = 0, 0, 0
        
        # Draw stats
        if SHOW_STATS:
            draw_text(disp_img, 'frame read time: %.1f ms' % (1000+display_frt), (20, 20))
            draw_text(disp_img, 'face detect time: %.1f ms' % (1000+display_fdt), (20, 40))
            draw_text(disp_img, 'fps: %.1f' % (1000+display_fps), (20, 60))
            draw_text(disp_img, 'time: %.3f' % (1000+time.time()), (20, 80))
            draw_text(disp_img, 'Total time: %.1f ms' % (1000+(1000*(t4-t1))), (20, 100))
        
        # Show images
        if SHOW:
            gray = cv2.flip(gray, 1)
            disp_img = cv2.flip(disp_img, 1)
            cv2.imshow('Grayscale', gray)
            cv2.imshow('Face Detector', disp_img)
            
        # Display stats on console
        print str(i) + " - Faces: " + str(faces_found) + " -- Eyes: " + str(eyes_found)
        
        # Exit on ESC
        # The waitKey function is very necessary.
        # Besides waiting the specified seconds for a key press (and returning -1 if none),
        # it also handles any windowing events, such as creating windows with imshow()
        # http://stackoverflow.com/questions/5217519/opencv-cvwaitkey
        if cv2.waitKey(5) == 27:
            break
    
    print (average/NUMBER_OF_TESTS)
            
