import pyrealsense2 as rs
import cv2
import numpy as np
import time
import pyximport; pyximport.install()
import substitute


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

start_time_current = time.time()
start_time_old = time.time()

print(start_time_old)
try:
    while True:
        time_at_start = time.time()

        start_time_old = start_time_current
        start_time_current = time.time()

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        coverage = depth_image.copy()

        # for y in range(480):
        #     for x in range(640):
        #         dist = depth_frame.get_distance(x, y)
        #         if dist > 1:
        #             coverage[y][x] = 9000

        coverage = substitute.substitute_distant_pixels(coverage, depth_frame)

        coverage = np.asanyarray(coverage)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        coverage_colormap = cv2.applyColorMap(cv2.convertScaleAbs(coverage, alpha=0.03), cv2.COLORMAP_JET)

        fps = 1 / (0.000001 + start_time_current - start_time_old)
        stats = "Output FPS: {}".format(int(fps))

        cv2.rectangle(color_image, (0, 0), (300, 25), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(color_image, stats, (6, 19), font, 0.5, (255, 255, 255), 1)

        images = np.hstack((color_image, depth_colormap, coverage_colormap))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    pipeline.stop()
