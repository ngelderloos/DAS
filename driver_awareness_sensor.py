import numpy as np
import cv2
import cv2.cv as cv
import time
import numpy as np

face_cascade_fn = "./data/haarcascades/haarcascade_frontalface_alt.xml"
eye_cascade_fn = "./data/haarcascades/haarcascade_eye.xml"

SMOOTH_FACE_MAX = 10
smooth_face_history = np.zeros((SMOOTH_FACE_MAX, 4), dtype=np.int32)
smooth_face_counter = 0

SHOW_REAL_TIME = False
SHOW_SMOOTH = True

#-------------------------------------------------------------------------
# detect_face() uses the given cascade to find the desired features.
# Receive: img, an image
#          cascade, the cascade to use to locate the features
# Return: rects, a list of rectangles, each represented as numpy.ndarray
#-------------------------------------------------------------------------
def detect_face(img, cascade):
    
    # detectMultiScale(image, rejectLevels, levelWeights[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]
    # image - Matrix of the type CV_8U containing an image where objects are detected.
    # rejectLevels - 
    # levelWeights - 
    # scaleFactor - Parameter specifying how mucht the image size is reduced at each image object.
    # minNeighbors - Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # flags - Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
    # minSize - Minimum possible object size. Objects smaller than that are ignored.
    # maxSize - Maximum possible object size. Objects larger than that are ignored.
    # outputRejectLevels - 
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100), flags = cv.CV_HAAR_SCALE_IMAGE)
    
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def detect_eyes(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50), flags=cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

#-------------------------------------------------------------------------
# draw_rects() draws a rectangle on the given image using the coordinates
# provided and line width of two pixels.
# Receive: img, an image
#          rects, a list of rectangles, each represented as numpy.ndarray
#          color, the color of the rectangle
#-------------------------------------------------------------------------
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#-------------------------------------------------------------------------
# draw_text() draws the given text on the image at the specified
# coordinates.
# Receive: img, an image
#          text, the text to draw on the image
#          (x, y), the coordinates of the lower left corner
#-------------------------------------------------------------------------
def draw_text(img, text, (x, y)):
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2, cv2.CV_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1, cv2.CV_AA)
    
def smooth_face(rects):
    global smooth_face_counter, smooth_face_history
    
    # choose which face rect to process
    # currently the first one is choosen, even if incorrect
    if len(rects) > 0:
        # store first rect in history array
        smooth_face_history[smooth_face_counter] = rects[0:1]
        smooth_face_counter += 1
        
        # reset counter, if necessary
        if smooth_face_counter >= SMOOTH_FACE_MAX:
            smooth_face_counter = 0
    
    x1sum = y1sum = x2sum = y2sum = 0
    
    for x1, y1, x2, y2 in smooth_face_history:
        x1sum += x1
        y1sum += y1
        x2sum += x2
        y2sum += y2
    
    return [np.array([int(x1sum/SMOOTH_FACE_MAX), int(y1sum/SMOOTH_FACE_MAX), int(x2sum/SMOOTH_FACE_MAX), int(y2sum/SMOOTH_FACE_MAX)])]

if __name__ == '__main__':
    
    # create cascades for face and eyes
    face_cascade = cv2.CascadeClassifier(face_cascade_fn)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_fn)
    
    # create video capture
    cam = cv2.VideoCapture(-1)
    
    
    frame_read_time, face_detect_time, frames_per_second = 0, 0, 0
    display_frt, display_fdt, display_fps = 0, 0, 0
    counter = 0
    previous_display_time = 0
    
    color = 0
    
    while True:
        t1 = time.time()
        
        # read image from camera
        ret, img = cam.read()
        
        # create grayscale image for faster processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        t2 = time.time()
        
        # detect faces in image
        face_rects = detect_face(gray, face_cascade)
        
        disp_img = img.copy()
        
        # show each detected face in real time
        if SHOW_REAL_TIME:
            draw_rects(disp_img, face_rects, (0, 255, 0))
        
        # if len(rects) > 1:
            # rect = rects[0:1]
            # print "---"
            # for item in rect:
                # print item
        
        # update smooth_face rectangle
        smoothed_face = smooth_face(face_rects)
        
        for x1, y1, x2, y2 in smoothed_face:
        # for x1, y1, x2, y2 in rects:
            face_area = gray[y1:y2, x1:x2]
            disp_img_face_area = disp_img[y1:y2, x1:x2]
            eye_rects = detect_eyes(face_area.copy(), eye_cascade)
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
            counter = 0
            frame_read_time, face_detect_time, frames_per_second = 0, 0, 0
        
        # Draw stats
        draw_text(disp_img, 'frame read time: %.1f ms' % (1000+display_frt), (20, 20))
        draw_text(disp_img, 'face detect time: %.1f ms' % (1000+display_fdt), (20, 40))
        draw_text(disp_img, 'fps: %.1f' % (1000+display_fps), (20, 60))
        draw_text(disp_img, 'time: %.3f' % (1000+time.time()), (20, 80))
        draw_text(disp_img, 'Total time: %.1f ms' % (1000+(1000*(t4-t1))), (20, 100))
        
        # Show images
        cv2.imshow('Grayscale', gray)
        cv2.imshow('Face Detector', disp_img)
        
        # Exit on ESC
        if cv2.waitKey(5) == 27:
            break
