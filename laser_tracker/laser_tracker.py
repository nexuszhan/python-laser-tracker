#! /usr/bin/env python
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

class LaserTracker(object):

    def __init__(self, cam_width=640, cam_height=480, hue_min=20, hue_max=160,
                 sat_min=100, sat_max=255, val_min=200, val_max=256,
                 display_thresholds=False):
        """
        * ``cam_width`` x ``cam_height`` -- This should be the size of the
        image coming from the camera. Default is 640x480.

        HSV color space Threshold values for a RED laser pointer are determined
        by:

        * ``hue_min``, ``hue_max`` -- Min/Max allowed Hue values
        * ``sat_min``, ``sat_max`` -- Min/Max allowed Saturation values
        * ``val_min``, ``val_max`` -- Min/Max allowed pixel values

        If the dot from the laser pointer doesn't fall within these values, it
        will be ignored.

        * ``display_thresholds`` -- if True, additional windows will display
          values for threshold image channels.

        """

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.display_thresholds = display_thresholds

        self.capture = None  # camera capture device
        self.pipeline = None
        self.config = None
        self.pipeline_wrapper = None
        self.pipeline_profile = None
        self.depth_scale = None
        
        self.channels = {
            'hue': None,
            'saturation': None,
            'value': None,
            'laser': None,
        }

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

    def setup_camera_capture(self, device_num=0):
        """Perform camera setup for the device number (default device = 0).
        Returns a reference to the camera Capture object.

        """
        # try:
        #     device = int(device_num)
        #     sys.stdout.write("Using Camera Device: {0}\n".format(device))
        # except (IndexError, ValueError):
        #     # assume we want the 1st device
        #     device = 0
        #     sys.stderr.write("Invalid Device. Using default device 0\n")

        # Try to start capturing frames
        # self.capture = cv2.VideoCapture(device)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        self.pipeline_profile = config.resolve(pipeline_wrapper)
        self.capture = pipeline_profile.get_device()
        self.config.enable_stream(rs.stream.depth, self.cam_width, self.cam_height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.cam_width, self.cam_height, rs.format.bgr8, 30)

        # if not self.capture.isOpened():
        #     sys.stderr.write("Faled to Open Capture device. Quitting.\n")
        #     sys.exit(1)

        # set the wanted image size from the camera
        # self.capture.set(
        #     cv2.cv.CV_CAP_PROP_FRAME_WIDTH if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_WIDTH,
        #     self.cam_width
        # )
        # self.capture.set(
        #     cv2.cv.CV_CAP_PROP_FRAME_HEIGHT if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_HEIGHT,
        #     self.cam_height
        # )
        return self.capture
    
    def clear_trail(self):
        self.trail = np.zeros((self.cam_height, self.cam_width, 3), np.uint8)
        self.centers.clear()

    def handle_quit(self, delay=10):
        """Quit the program if the user presses "Esc" or "q"."""
        key = cv2.waitKey(delay)
        c = chr(key & 255)
        if c in ['c', 'C']:
            self.trail = np.zeros((self.cam_height, self.cam_width, 3),
                                     np.uint8)
        if c in ['q', 'Q', chr(27)]:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            sys.exit(0)

    def threshold_image(self, channel):
        if channel == "hue":
            minimum = self.hue_min
            maximum = self.hue_max
        elif channel == "saturation":
            minimum = self.sat_min
            maximum = self.sat_max
        elif channel == "value":
            minimum = self.val_min
            maximum = self.val_max

        (t, tmp) = cv2.threshold(
            self.channels[channel],  # src
            maximum,  # threshold value
            0,  # we dont care because of the selected type
            cv2.THRESH_TOZERO_INV  # t type
        )

        (t, self.channels[channel]) = cv2.threshold(
            tmp,  # src
            minimum,  # threshold value
            255,  # maxvalue
            cv2.THRESH_BINARY  # type
        )

        if channel == 'hue':
            # only works for filtering red color because the range for the hue
            # is split
            self.channels['hue'] = cv2.bitwise_not(self.channels['hue'])

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

        # split the video frame into color channels
        h, s, v = cv2.split(hsv_img)
        self.channels['hue'] = h
        self.channels['saturation'] = s
        self.channels['value'] = v

        # Threshold ranges of HSV components; storing the results in place
        self.threshold_image("hue")
        self.threshold_image("saturation")
        self.threshold_image("value")

        # Perform an AND on HSV components to identify the laser!
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue'],
            self.channels['value']
        )
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['saturation'],
            self.channels['laser']
        )

        # Merge the HSV components back together.
        hsv_image = cv2.merge([
            self.channels['hue'],
            self.channels['saturation'],
            self.channels['value'],
        ])

        self.track(frame, self.channels['laser'])

        return hsv_image

    def display(self, img, frame):
        """Display the combined image and (optionally) all other image channels
        NOTE: default color space in OpenCV is BGR.
        """
        cv2.imshow('RGB_VideoFrame', frame)
        cv2.imshow('LaserPointer', self.channels['laser'])
        if self.display_thresholds:
            cv2.imshow('Thresholded_HSV_Image', img)
            cv2.imshow('Hue', self.channels['hue'])
            cv2.imshow('Saturation', self.channels['saturation'])
            cv2.imshow('Value', self.channels['value'])

    def setup_windows(self):
        sys.stdout.write("Using OpenCV version: {0}\n".format(cv2.__version__))

        # create output windows
        self.create_and_position_window('LaserPointer', 0, 0)
        self.create_and_position_window('RGB_VideoFrame',
                                        10 + self.cam_width, 0)
        if self.display_thresholds:
            self.create_and_position_window('Thresholded_HSV_Image', 10, 10)
            self.create_and_position_window('Hue', 20, 20)
            self.create_and_position_window('Saturation', 30, 30)
            self.create_and_position_window('Value', 40, 40)

    def get_distance(frames):
        """compute distance to a laser point"""
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        aligned_depth_frame = frames.get_depth_frame()

        depth = np.asanyarray(aligned_depth_frame.get_data())
        # depth = depth[xmin_depth:xmax_depth,ymin_depth:ymax_depth].astype(float)
        # depth = depth * depth_scale
        # dist,_,_,_ = cv2.mean(depth)
    
    def setup(self):
        # Set up window positions
        self.setup_windows()
        # Set up the camera capture
        self.setup_camera_capture()

        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def run(self):
        # Set up window positions
        self.setup_windows()
        # Set up the camera capture
        self.setup_camera_capture()

        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        while True:
            # 1. capture the current image
            # success, frame = self.capture.read()
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # if not success:  # no image captured... end the processing
            #     sys.stderr.write("Could not read camera frame. Quitting\n")
            #     sys.exit(1)
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # hsv_image = self.detect(frame)
            # self.display(hsv_image, frame)
            hsv_image = self.detect(color_image)
            self.display(hsv_image, color_image)
            self.handle_quit()

class Runner():
    def __init__(self, params):
        self.tracker = LaserTracker(
            cam_width=params.width,
            cam_height=params.height,
            hue_min=params.huemin,
            hue_max=params.huemax,
            sat_min=params.satmin,
            sat_max=params.satmax,
            val_min=params.valmin,
            val_max=params.valmax,
            display_thresholds=params.display
        )
        self.state = States.IDLE
        self.target_pos = None

    def handle_idle(self):
        while self.state == States.IDLE:
            frames = self.tracker.pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            hsv_image = self.tracker.detect(color_image)
            self.tracker.display(hsv_image, color_image)
            # if ... :
                # self.state = States.SELECTED

            self.tracker.handle_quit()

    def handle_selected(self):
        while self.state == States.SELECTED:
            frames = self.tracker.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            hsv_image = self.tracker.detect(color_image)
            self.tracker.display(hsv_image, color_image)
            self.tracker.handle_quit()

    def start(self):
        self.tracker.setup()
        while True:
            self.tracker.clear_trail()
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
    parser.add_argument('-W', '--width',
                        default=640,
                        type=int,
                        help='Camera Width')
    parser.add_argument('-H', '--height',
                        default=480,
                        type=int,
                        help='Camera Height')
    parser.add_argument('-u', '--huemin',
                        default=20,
                        type=int,
                        help='Hue Minimum Threshold')
    parser.add_argument('-U', '--huemax',
                        default=160,
                        type=int,
                        help='Hue Maximum Threshold')
    parser.add_argument('-s', '--satmin',
                        default=100,
                        type=int,
                        help='Saturation Minimum Threshold')
    parser.add_argument('-S', '--satmax',
                        default=255,
                        type=int,
                        help='Saturation Maximum Threshold')
    parser.add_argument('-v', '--valmin',
                        default=200,
                        type=int,
                        help='Value Minimum Threshold')
    parser.add_argument('-V', '--valmax',
                        default=255,
                        type=int,
                        help='Value Maximum Threshold')
    parser.add_argument('-d', '--display',
                        action='store_true',
                        help='Display Threshold Windows')
    params = parser.parse_args()

    runner = Runner(params)
    runner.start()
