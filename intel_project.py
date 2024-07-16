#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageListener(Node):

    def __init__(self):
        super().__init__('image_listener')
        self.bridge = CvBridge()
        self.images = {
            1: None,
            2: None,
            3: None,
            4: None
        }

        self.create_subscription(Image, '/overhead_camera/overhead_camera1/image_raw',
                                 self.listener_callback1, 10)
        self.create_subscription(Image, '/overhead_camera/overhead_camera2/image_raw',
                                 self.listener_callback2, 10)
        self.create_subscription(Image, '/overhead_camera/overhead_camera3/image_raw',
                                 self.listener_callback3, 10)
        self.create_subscription(Image, '/overhead_camera/overhead_camera4/image_raw',
                                 self.listener_callback4, 10)

    def listener_callback1(self, msg):
        self.images[1] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.stitch_images()

    def listener_callback2(self, msg):
        self.images[2] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.stitch_images()

    def listener_callback3(self, msg):
        self.images[3] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.stitch_images()

    def listener_callback4(self, msg):
        self.images[4] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.stitch_images()

    def stitch_images(self):
        if all(img is not None for img in self.images.values()):
            # Resize images (optional)
            for key in self.images:
                self.images[key] = cv2.resize(self.images[key], (640, 480))

            # Define the order of stitching
            order = [[4, 3], [2, 1]]

            # Initialize a list to hold images in the correct order
            ordered_images = []

            # Arrange images according to order
            for row in order:
                row_images = [self.images[col] for col in row]
                ordered_images.append(row_images)

            # Create a list to hold stitched images
            stitched_images = []

            # Iterate over ordered pairs of images and stitch them
            for i in range(len(ordered_images)):
                # Stitch images horizontally
                stitched_row = np.hstack(ordered_images[i])
                stitched_images.append(stitched_row)

            # Stitch the rows vertically to form the final composite image
            final_image = np.vstack(stitched_images)

            # Display the final image
            cv2.imshow("Stitched Image", final_image)
            cv2.waitKey(1)

            # Create occupancy map
            occupancy_map = self.create_occupancy_map(final_image)

            # Process occupancy map (example: print occupancy grid)
            print("Occupancy Map:")
            print(occupancy_map)

    def create_occupancy_map(self, image):
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresholded = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

        # Invert thresholded image (optional, depending on how you want to define occupancy)
        thresholded = cv2.bitwise_not(thresholded)

        # Create occupancy grid (0 for free space, 1 for occupied space)
        occupancy_grid = thresholded / 255

        return occupancy_grid

def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

