import sys
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs

from enum import Enum

class States(Enum):
    IDLE = 1
    SELECTED = 2
    NAVIGATING = 3
    ARRIVE = 4        

class LaserTracker():
    def __init__(self, hsv_thres, color, cam_width, cam_height, xpos=0, ypos=0):
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.xpos = xpos
        self.ypos = ypos

        self.hue_min = hsv_thres["hue_min"]
        self.hue_max = hsv_thres["hue_max"]
        self.sat_min = hsv_thres["sat_min"]
        self.sat_max = hsv_thres["sat_max"]
        self.val_min = hsv_thres["val_min"]
        self.val_max = hsv_thres["val_max"]
        self.color = color

        self.previous_position = None
        self.trail = np.zeros((self.cam_height, self.cam_width, 3),
                                 np.uint8)
        self.centers = []
    
    def create_and_position_window(self, name, xpos, ypos):
        """Creates a named widow placing it on the screen at (xpos, ypos)."""
        # Create a window
        cv2.namedWindow(name)
        # Resize it to the size of the camera image
        cv2.resizeWindow(name, self.cam_width, self.cam_height)
        # Move to (xpos,ypos) on the screen
        cv2.moveWindow(name, xpos, ypos)

    def display(self, thres, frame):
        """Display the combined image and (optionally) all other image channels
        NOTE: default color space in OpenCV is BGR.
        """
        cv2.imshow('RGB_VideoFrame', frame)
        cv2.imshow('LaserPointer', thres)

    def setup_windows(self):
        sys.stdout.write("Using OpenCV version: {0}\n".format(cv2.__version__))

        # create output windows
        self.create_and_position_window('LaserPointer', self.xpos, self.ypos)
        self.create_and_position_window('RGB_VideoFrame',
                                        self.xpos + 10 + self.cam_width, self.ypos)

    def clear_trail(self):
        self.trail = np.zeros((self.cam_height, self.cam_width, 3), np.uint8)
        self.centers.clear()
        self.previous_position = None

    def track(self, frame, mask):
        """
        Track the position of the laser pointer.

        Code taken from
        http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
        """
        center = None

        countours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(countours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(countours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            moments = cv2.moments(c)
            if moments["m00"] > 0:
                center = int(moments["m10"] / moments["m00"]), \
                         int(moments["m01"] / moments["m00"])
            else:
                center = int(x), int(y)

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                # then update the ponter trail
                if self.previous_position:
                    cv2.line(self.trail, self.previous_position, center,
                             (255, 255, 255), 2)

        cv2.add(self.trail, frame, frame)
        self.previous_position = center
        self.centers.append(center)

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold ranges of HSV components
        hsv_thres = cv2.inRange(hsv_img, np.array(self.hue_min, self.sat_min, self.val_min), 
                                    np.array(self.hue_max, self.sat_max, self.val_max))
        if self.color == "red":
            tmp = cv2.inRange(hsv_img, np.array(170, self.sat_min, self.val_min),
                              np.array(180, self.sat_max, self.val_max))
            hsv_thres = cv2.bitwise_or(hsv_thres, tmp)

        self.track(frame, hsv_thres)

        return hsv_thres
    
    def check_activate(self):
        """Check whether there is a clockwise circle"""
        return True

    def get_target_pos(self):
        """Identify the target position when selected"""
        pos = None
        pass

class Runner():
    def __init__(self, display=1):
        self.display = display
        self.cam_width = 640
        self.cam_height = 480
        self.hsv_thres = {
            "red":{"hue_min":0, "hue_max":10, "sat_min":100, 
                   "sat_max":255, "val_min":200, "val_max":256}, 
            "green":{"hue_min":50, "hue_max":70, "sat_min":100, 
                   "sat_max":255, "val_min":200, "val_max":256}}
        self.red_tracker = LaserTracker(self.hsv_thres["red"], "red", self.cam_width, 
                                        self.cam_height)
        self.green_tracker = LaserTracker(self.hsv_thres["green"], "green", self.cam_width,
                                          self.cam_height, 0, self.cam_height+10)

        self.pipeline = None

        self.state = States.IDLE
        self.target_pos = None

    def run_camera(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.capture = self.pipeline_profile.get_device()
        self.config.enable_stream(rs.stream.depth, self.cam_width, self.cam_height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.cam_width, self.cam_height, rs.format.bgr8, 30)

        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
    def handle_quit(self, delay=10):
        """Quit the program if the user presses "Esc" or "q"."""
        key = cv2.waitKey(delay)
        c = chr(key & 255)
        if c in ['c', 'C']:
            self.red_tracker.clear_trail()
            self.green_tracker.clear_trail()
        if c in ['s', 'S']:
            self.state = States.SELECTED
        if c in ['i', 'I']:
            self.state = States.IDLE
        if c in ['q', 'Q', chr(27)]:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            sys.exit(0)
    
    def handle_idle(self):
        while self.state == States.IDLE:
            frames = self.pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            red_bin = self.red_tracker.detect(color_image, "red")
            green_bin = self.green_tracker.detect(color_image, "green")
            if self.display:
                self.red_tracker.display(red_bin, color_image)
                self.green_tracker.display(green_bin, color_image)

            if self.red_tracker.check_activate():
                self.state = States.SELECTED

            self.handle_quit()

    def handle_selected(self):
        while self.state == States.SELECTED:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            red_bin = self.red_tracker.detect(color_image, "red")
            green_bin = self.green_tracker.detect(color_image, "green")
            # self.red_tracker.get_target_pos()
            # self.green_tracker.get_target_pos()
            if self.display:
                self.red_tracker.display(red_bin, images)
                self.green_tracker.display(green_bin, images)
            self.handle_quit()

    def start(self):
        self.run_camera()
        while True:
            self.red_tracker.clear_trail()
            self.green_tracker.clear_trail()
            if self.state == States.IDLE:
                self.handle_idle()
            elif self.state == States.SELECTED:
                self.handle_selected()
            elif self.state == States.NAVIGATING:
                pass
            elif self.state == States.ARRIVE:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Laser Tracker')
    parser.add_argument('-d', '--display',
                        default=1,
                        type=int,
                        help='Display the video or not')
    params = parser.parse_args()
    runner = Runner(params.display)
    runner.start()
    