import pyrealsense2 as rs
import cv2
import numpy as np
import time
import concurrent.futures
import logging

class RealsenseFrameService:

    # adjust the logging level here (e.g debug or info etc.)
    logging.basicConfig(level = logging.DEBUG)
    logger = logging.getLogger("realsense-frame-service")

    offset_x = 20
    offset_y = -20
    grey_color = 153

    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        self.logger.debug(f"Depth Scale is: {depth_scale}")

        clipping_distance_in_meters = 1  # in meters
        self.clipping_distance = clipping_distance_in_meters / depth_scale
        self.align = rs.align(rs.stream.color)

    def fetch_frames(self, align):
        
        tic = time.time()
        frames = self.pipeline.wait_for_frames()
        toc = time.time()
        self.logger.debug(f"Time for waiting for next frame: {toc - tic:0.4f} seconds")

        with concurrent.futures.ThreadPoolExecutor() as executor:

            color_depth_frame_future = executor.submit(self.__get_color_depth_frame, frames)

            segmented_frame = None
            if align:
                segemented_frame_future = executor.submit(self.__get_segmented_frame, frames)
                (segmented_frame) = segemented_frame_future.result()

            (color_frame, depth_frame) = color_depth_frame_future.result()

            return color_frame, depth_frame, segmented_frame

    
    def __segment_color_frame(self, depth_image, color_image):

        # Remove background - Set pixels further than clipping_distance to grey
        tic = time.time()
        segmented_image = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), self.grey_color, color_image)
        toc = time.time()
        self.logger.debug(f"Time for filtering: {toc - tic:0.4f} seconds")

        return segmented_frame

    def __get_depth_color_from_frames(self, frames):

        tic = time.time()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        toc = time.time()
        self.logger.debug(f"Time for getting frames: {toc - tic:0.4f} seconds")

        return depth_frame, color_frame

    def __frames_to_arrays(self, depth_image, color_frame):

        tic = time.time()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        toc = time.time()
        self.logger.debug(f"Overall time for array creation: {toc - tic:0.4f} seconds")+

        return depth_frame, color_frame

    def __get_segmented_frame(self, frames):

        tic = time.time()
        aligned_frames = self.align.process(frames)
        toc = time.time()
        self.logger.debug(f"Time for aligning frames: {toc - tic:0.4f} seconds")

        (depth_frame, depth_frame) = self.__get_depth_color_from_frames(aligned_frames)

        if not depth_frame or not color_frame:
            return

        depth_frame, color_frame = self.__frames_to_arrays(depth_frame, color_frame)

        tic = time.time()
        depth_image_3d = np.dstack(
            (depth_frame, depth_frame, depth_frame))  # depth image is 1 channel, color is 3 channels
        toc = time.time()
        self.logger.debug(f"Time for stacking: {toc - tic:0.4f} seconds")

        return self.__segment_color_frame(depth_image_3d, color_frame)

    def __get_color_depth_frame(self, frames):

        depth_frame, color_frame = self.__get_depth_color_from_frames(frames)

        if not depth_frame or not color_frame:
            return
        
        depth_frame, color_frame = self.__frames_to_arrays(depth_frame, color_frame)

        return color_frame, depth_frame

    def stop_pipeline(self):
        self.pipeline.stop()

