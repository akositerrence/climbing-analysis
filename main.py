import numpy as np
import cv2 as cv

class BoardDetector():
    def __init__(self, ref_image, lowes_test_ratio=0.65, minimum_match_count = 50, sift_frame_rate = 30, board_height = 3.26, board_width = 2.44):
        self.reference_image_file = ref_image
        self.ratio = lowes_test_ratio
        self.min_match_count = minimum_match_count
        self.sift_fr = sift_frame_rate
        self.board_h = board_height
        self.board_w = board_width
        self.last_polygon = None
        self.sift_object = cv.SIFT_create()
        self.feature_matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        
        self.board_corners_matrix = np.array([
            [0.0,       0.0,       0.0],
            [self.board_w, 0.0,       0.0],
            [self.board_w, self.board_h, 0.0],
            [0.0,       self.board_h, 0.0],
        ])
        self.three_dimensional_points = []
        self.two_dimensional_points = []
        self.img_shape = None
        
        self.K = None
        self.dist = None
        
    def initialize(self):
        capture = cv.VideoCapture(1)
        if not capture.isOpened():
            print("... Camera unavailable ...")
            exit()
        return capture

    def order_quadrilateral(self, points):
        points = np.asarray(points)
        sum = points[:, 0] + points[:, 1]
        difference = points[:, 0] - points[:, 1]
        
        top_left = points[np.argmin(sum)] 
        bottom_right = points[np.argmax(sum)] 
        top_right = points[np.argmin(difference)] 
        bottom_left = points[np.argmax(difference)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left])
        
    #####################################################################
    
    def line_through(self, p1, p2):
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        return np.cross([x1, y1, 1.0], [x2, y2, 1.0])

    def intersect(self, l1, l2):
        p = np.cross(l1, l2)
        return np.array([p[0] / p[2], p[1] / p[2]])
    
    def calibrate_from_first_polygon(self, frame_shape, polygon_4x1x2):
        h, w = frame_shape[:2]
        cx, cy = w / 2.0, h / 2.0
        c = np.array([cx, cy], dtype=np.float64)

        tl, tr, br, bl = self.order_quad(polygon_4x1x2)

        l_top = self.line_through(tl, tr)
        l_bot = self.line_through(bl, br)
        vp_h = self.intersect(l_top, l_bot)

        l_left  = self.line_through(tl, bl)
        l_right = self.line_through(tr, br)
        vp_v = self.intersect(l_left, l_right)

        f2 = -np.dot(vp_h - c, vp_v - c)
        assert f2 > 0.0, "... Need a tilted view of the board (vanishing points must be finite) ..."
        f = float(np.sqrt(f2))

        self.K = np.array([[f, 0.0, cx],
                        [0.0, f, cy],
                        [0.0, 0.0, 1.0]], dtype=np.float64)

        self.dist = np.zeros((5, 1), dtype=np.float64)

    def pose_from_polygon(self, frame, polygon_4x1x2):
        obj = np.array([
            [0.0,         0.0,         0.0],
            [self.board_w, 0.0,         0.0],
            [self.board_w, self.board_h, 0.0],
            [0.0,         self.board_h, 0.0],
        ], dtype=np.float32)

        img = self.order_quad(polygon_4x1x2).astype(np.float32) 

        ok, rvec, tvec = cv.solvePnP(obj, img, self.K, self.dist, flags=cv.SOLVEPNP_ITERATIVE)
        if ok:
            cv.drawFrameAxes(frame, self.K, self.dist, rvec, tvec, 0.25)  
        return ok
    
    #####################################################################

    def camera_metrics(self, focal_length_x, focal_length_y, principal_point_x, principal_point_y, dist):
        K = np.array([[focal_length_x, 0.0, principal_point_x],
                  [0.0, focal_length_y, principal_point_y],
                  [0.0, 0.0, 1.0]])
        dist = np.asarray(dist).reshape(-1, 1)
        K = [[focal_length_x, 0, principal_point_x], [0, focal_length_y, principal_point_y], [0, 0, 1 ]]
        return K, dist
        
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
        