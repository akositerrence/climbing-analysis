import numpy as np
import cv2 as cv

class BoardDetector():
    def __init__(self, ref_image, lowes_test_ratio=0.65, minimum_match_count = 50, sift_frame_rate = 30):
        self.reference_image_file = ref_image
        self.ratio = lowes_test_ratio
        self.min_match_count = minimum_match_count
        self.sift_fr = sift_frame_rate
        self.last_polygon = None
        self.sift_object = cv.SIFT_create()
        self.feature_matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        
    def initialize(self):
        capture = cv.VideoCapture(1)
        if not capture.isOpened():
            print("... Camera unavailable ...")
            exit()
        return capture

    def apply_grayscale(self, frame):
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
        
    def detect_board(self):
        capture = self.initialize()
        reference_image = self.apply_grayscale(cv.imread(self.reference_image_file))
        keypoints_nominal, descriptors_nominal = self.sift_object.detectAndCompute(reference_image ,None)
        frame_counter = 0
        
        while True:
            ret, frame = capture.read()
            if ret == False:
                print("... Exiting ...")
                break
            grayscale_frame = self.apply_grayscale(frame)
            frame_counter = frame_counter + 1
            if ((frame_counter % self.sift_fr) == 0):
                keypoints_seen, descriptors_seen = self.sift_object.detectAndCompute(grayscale_frame, None)
                if descriptors_seen is None or descriptors_nominal is None:
                    continue
                matches = self.feature_matcher.knnMatch(descriptors_nominal, descriptors_seen, k=2)
                
                good_matches = []
                for m, n in matches:
                    if m.distance < self.ratio * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) > self.min_match_count:
                    source_points = np.float32([ keypoints_nominal[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    destination_points = np.float32([ keypoints_seen[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    
                    M, mask = cv.findHomography(source_points, destination_points, cv.RANSAC, 5.0)
                    
                    h,w = reference_image.shape
                    points = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    destination = cv.perspectiveTransform(points,M)
                    self.last_polygon = np.int32(destination)
                    frame = cv.polylines(frame, [self.last_polygon], True, 255, 3, cv.LINE_AA)
                
            if self.last_polygon is not None:
                    cv.polylines(frame, [self.last_polygon], True, 255, 3, cv.LINE_AA)
                    
                ###
                #matches = sorted(good_matches, key = lambda x:x.distance)           
                #matched_img = cv.drawMatches(reference_image, keypoints_nominal, grayscale_frame, keypoints_seen, good_matches[:50], None, flags=2)         
                #cv.imwrite("test.jpg", matched_img)
                ###
                
            cv.imshow("frame", frame)
            if cv.waitKey(1) == ord('q'):
                break
            
        self.exit_processes(capture)
            
    def exit_processes(self, camera_object):
        camera_object.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    detector = BoardDetector("moonboard.png")
    detector.detect_board()
        