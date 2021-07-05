#######################################################
#  THIS CODE HAS BEEN OMITTED DUE TO CONFIDENTIALITY  #
#  DO NOT RUN THIS CODE.                              #
#######################################################

import CameraSDK as cs  # NOT REAL NAME
from detect_rect import DetectRect
import numpy as np
import cv2

# Predefine some values of variables. FROM JSON FILE
widthPixels = 1280
heightPixels = 720
laserPower = 55
confidenceThreshold = 2
minDistance = 0
receiverGain = 18
postProcessing = 2
preProcessing = 5
NoiseFiltering = 6

# Configure depth and color streams
pipeline = cs.pipeline()
config = cs.config()
colorizer = cs.colorizer()
config.enable_all_streams

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see cs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Set depth level controls
depth_sensor.set_option(minDistance, laserPower,
                        receiverGain, confidenceThreshold, NoiseFiltering)

# 0=Dynamic, 1=Fixed, 2=Near, 3=Far
# Set color scheme, and coloring options
colorizer.set_option(0, 0.15, 2.3, 2)

# Create an align object
# rs.align allows us to perform alignment of depth frames to othecs frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = cs.stream.color
align = cs.align(align_to)

depth_profile = cs.video_stream_profile(profile.get_stream(cs.stream.depth))
intrinsics = depth_profile.get_intrinsics()
print(intrinsics)

color_profile = cs.video_stream_profile(profile.get_stream(cs.stream.color))


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        depth_colorized = np.asanyarray(
            colorizer.colorize(aligned_depth_frame).get_data()
        )
        color_image = np.asanyarray(color_frame.get_data())

        # Apply color scheme corrections
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Develop rectangle in videos image processing methods
        det_rect = DetectRect()
        x, y, w, h, rects = det_rect.detect(color_image)
        image_rect = color_image
        grayed = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(grayed, (7, 7), 0)
        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((2, 2), np.uint8)
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        thresh_img = cv2.threshold(gradient, 15, 225, cv2.THRESH_BINARY)[1]
        dilated = cv2.dilate(thresh_img, kernel2, iterations=2)

        cv2.namedWindow("DEPTH", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("DEPTH", dilated)  # image_rect
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()
