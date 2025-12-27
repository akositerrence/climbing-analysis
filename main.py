import numpy as np
import cv2 as cv

def initialize():
    capture = cv.VideoCapture(1)
    if not capture.isOpened():
        print("... Camera unavailable ...")
        exit()
    return capture

def detect_board():
    capture = initialize()
    while True:
        ret, frame = capture.read()
        if ret == False:
            print("... Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("frame", gray)
        if cv.waitKey(1) == ord('q'):
            break
        
def exit_processes(camera_object):
    camera_object.release()
    cv.destroyAllWindows()

detect_board()

    