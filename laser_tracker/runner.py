from laser_tracker import *
# from pid_controller import *
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, Pose, Point
# from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
# from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy

from enum import Enum

class States(Enum):
    """enums for states"""
    IDLE = 1
    SELECTED = 2
    NAVIGATING = 3
    ARRIVE = 4   

class Tasks(Enum):
    """enums for tasks"""
    SPIN = 1
    DRIVE_SQUARE = 2

class Runner():
    def __init__(self, display=False):
        self.display = display
        self.cam_width = 640
        self.cam_height = 480
        self.hsv_thres = {
            "red":{"hue_min":0, "hue_max":10, "sat_min":100, 
                   "sat_max":255, "val_min":200, "val_max":256}, 
            "green":{"hue_min":50, "hue_max":70, "sat_min":100, 
                   "sat_max":255, "val_min":200, "val_max":256}} # 100, 255
        self.red_tracker = LaserTracker(self.hsv_thres["red"], "red", self.cam_width, 
                                        self.cam_height)
        self.green_tracker = LaserTracker(self.hsv_thres["green"], "green", self.cam_width,
                                          self.cam_height, 0, self.cam_height+10)

        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = None
        # self.depth_intrin = None

        self.state = States.IDLE #States.IDLE
        self.task = None
        self.offset = [0.0, 0.0]

        # self.navigator = BasicNavigator()
        
        # self.target_pos = Point()
        # self.current_pos = Point()
        
        # self.target_pos_node = Node("target pos publisher")
        
        # self.odom_node = Node("odometry subscriber")
        
        # self.target_pos_publisher = self.target_pos_node.create_publisher(
        #     Odometry,
        #     "target_pos",
        #     10
        # )
        # self.odom_subscriber = self.odom_node.create_subscription(
        #     Odometry, 
        #     "odom",
        #     self.odom_callback,
        #     10)

        # # self.pid = PID(1, 0.1, 0.05, setpoint=1)
        # self.pid = MYPID(1.0, 0.1, 0.03)

    # def odom_callback(self, msg:Odometry):
    #     self.current_pos = msg.pose.pose.position

    def setup_camera(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.config.enable_stream(rs.stream.depth, self.cam_width, self.cam_height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.cam_width, self.cam_height, rs.format.bgr8, 30)

        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

        print("Camera setup succeed.")
        
    def handle_quit(self, delay=10):
        """Quit the program if the user presses "Esc" or "q"."""
        key = cv2.waitKey(delay)
        c = chr(key & 255)
        if c in ['c', 'C']:
            self.red_tracker.clear_trail()
            self.green_tracker.clear_trail()
        # if c in ['s', 'S']:
        #     self.state = States.SELECTED
        # if c in ['i', 'I']:
        #     self.state = States.IDLE
        if c in ['q', 'Q', chr(27)]:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            sys.exit(0)
    
    def handle_idle(self):
        while self.state == States.IDLE:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # red_bin = self.red_tracker.detect(color_image)
            green_bin = self.green_tracker.detect(color_image)
            if self.display:
                # self.red_tracker.display(red_bin, color_image)
                self.green_tracker.display(green_bin, color_image)
                self.handle_quit()

            # if self.red_tracker.check_activate():
            if self.green_tracker.check_activate():
                self.state = States.SELECTED

    def handle_selected(self):
        while self.state == States.SELECTED:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            # red_bin = self.red_tracker.detect(color_image)
            green_bin = self.green_tracker.detect(color_image)
            # print(self.green_tracker.centers)
            # red_pos = self.red_tracker.get_target_pos(depth_image, self.depth_scale)
            # print(red_pos)
            # if red_pos:
                # self.state = States.NAVIGATING
                # TODO: control ...
            
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            green_pos = self.green_tracker.get_target_pos(depth_image, self.depth_scale, depth_intrin)
            print(green_pos)
            if self.display:
                # self.red_tracker.display(red_bin, images)
                self.green_tracker.display(green_bin, images)
                self.handle_quit()

            if (green_pos):
                self.state = States.NAVIGATING
                self.target_pos_x = green_pos[0]
                self.target_pos_y = green_pos[1]
                # TODO: set the msg
                # msg = Odometry()
                # msg.header.stamp = self.target_pos_node.get_clock().now().to_msg()
                # msg.pose.pose.position = self.target_pos
                # self.target_pos_publisher.publish(msg)

    # def check_arrive(self):
    #     return abs(self.current_pos[0]-self.target_pos[0]) < 0.1 and \
    #             abs(self.current_pos[1]-self.target_pos[1]) < 0.1

    def handle_navigate(self):
        while self.state == States.NAVIGATING:
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            # goal_pose.header.stamp = navigator.get_clock().now().to_msg()
            goal_pose.pose.position.x = self.target_pos_x
            goal_pose.pose.position.y = self.target_pos_y
            goal_pose.pose.orientation.w = 1.0
            # navigator.goToPose(goal_pose)
            # while not navigator.isTaskComplete():
            #     continue
            
            # result = navigator.getResult()
            # if result == TaskResult.SUCCEEDED:
            #     print('Goal succeeded!')
            # elif result == TaskResult.CANCELED:
            #     print('Goal was canceled!')
            # elif result == TaskResult.FAILED:
            #     print('Goal failed!')
            # else:
            #     print('Goal has an invalid return status!')

            self.state = States.ARRIVE
            break

            # frames = self.pipeline.wait_for_frames()
            # aligned_frames = self.align.process(frames)
            # depth_frame = aligned_frames.get_depth_frame()
            # color_frame = frames.get_color_frame()

            # if not depth_frame or not color_frame:
            #     continue

            # depth_image = np.asanyarray(depth_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # images = np.hstack((color_image, depth_colormap))

            # # red_bin = self.red_tracker.detect(color_image)
            # green_bin = self.green_tracker.detect(color_image)
            # # print(self.green_tracker.centers)
            # # red_pos = self.red_tracker.get_target_pos(depth_image, self.depth_scale)
            # # print(red_pos)
            # # if red_pos:
            #     # self.state = States.NAVIGATING
            #     # TODO: control ...
            
            # depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            # green_pos = self.green_tracker.get_target_pos(depth_image, self.depth_scale, depth_intrin)
            # print(green_pos)
            # if self.display:
            #     # self.red_tracker.display(red_bin, images)
            #     self.green_tracker.display(green_bin, images)
            #     self.handle_quit()

            # if (green_pos):
            #     self.target_pos = green_pos

    def handle_arrive(self):
        # TODO: control the bot to do something
        while self.state == States.ARRIVE:
            if not self.task:
                self.state = States.IDLE
                break

            if self.task == Tasks.SPIN:
                pass
            elif self.task == Tasks.DRIVE_SQUARE:
                pass
            self.task = None
            self.state = States.IDLE
            
    def start(self):
        # rclpy.init()
        # rclpy.spin(self.target_pos_publisher)
        # rclpy.spin(self.odom_subscriber)

        if self.display:
            print("Display enabled")
        if self.display:
            # self.red_tracker.setup_windows()
            self.green_tracker.setup_windows()
        self.setup_camera()
        # TODO: initialize current_pos
        # self.current_pos = [0, 0]

        # Set nav2 initial pose
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        # initial_pose.header.stamp = navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = 0.0
        initial_pose.pose.position.y = 0.0
        initial_pose.pose.orientation.z = 0.0
        initial_pose.pose.orientation.w = 1.0
        # navigator.setInitialPose(initial_pose)

        # navigator.waitUntilNav2Active()

        while True:
            if self.state == States.IDLE:
                print("idel")
                self.handle_idle()
            elif self.state == States.SELECTED:
                print("selected")
                self.handle_selected()
            elif self.state == States.NAVIGATING:
                print("navigating")
                self.handle_navigate()
            elif self.state == States.ARRIVE:
                print("arrived    may perform some tasks...")
                self.handle_arrive()

            self.red_tracker.clear_trail()
            self.green_tracker.clear_trail()
            print("changing state")
            time.sleep(1)

        scv2.destroyAllWindows()
        self.pipeline.stop()
        rclpy.shutdown()
        # navigator.lifecycleShutdown()
        sys.exit(0)

if __name__ == '__main__':
    """run with 'python3 runner.py -d' to show video"""
    parser = argparse.ArgumentParser(description='Run the Laser Tracker')
    parser.add_argument('-d', '--display',
                        action='store_true',
                        help='Display the video or not')
    params = parser.parse_args()

    runner = Runner(params.display)
    runner.start()
    