import pyrealsense2 as rs
import cv2
import numpy as np
import time
import concurrent.futures

class RealsenseFrameService:

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
        print("Depth Scale is: ", depth_scale)

        clipping_distance_in_meters = 1  # 1 meter
        self.clipping_distance = clipping_distance_in_meters / depth_scale
        self.align = rs.align(rs.stream.color)

    def fetch_images(self, align):
        
        tic = time.time()
        frames = self.pipeline.wait_for_frames()
        toc = time.time()
        print(f"Time for waiting for next frame: {toc - tic:0.4f} seconds")

        with concurrent.futures.ThreadPoolExecutor() as executor:

            futureImages = executor.submit(self.get_images, frames)

            segmented_image = None
            if align:
                futureSegmentedImages = executor.submit(self.get_segmented_image, frames)
                (segmented_image) = futureSegmentedImages.result()

            (color_image, depth_image) = futureImages.result()


            return color_image, depth_image, segmented_image

    def get_segmented_image(self, frames):

            tic = time.time()
            aligned_frames = self.align.process(frames)
            toc = time.time()
            print(f"Time for aligning frames: {toc - tic:0.4f} seconds")

            tic = time.time()
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            toc = time.time()
            print(f"Time for getting frames: {toc - tic:0.4f} seconds")

            if not depth_frame or not color_frame:
                return

            tic = time.time()
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            toc = time.time()
            print(f"Overall time for array creation: {toc - tic:0.4f} seconds")

            # tic = time.time()
            # (image_x_size, image_y_size) = depth_frame.shape
            # aligned_depth_frame = np.zeros(depth_frame.shape)
            # aligned_depth_frame[0:image_x_size - self.offset_y, 0:image_y_size - self.offset_x] = depth_frame[self.offset_y:, self.offset_x:]
            # toc = time.time()
            # print(f"Time for self alignment: {toc - tic:0.4f} seconds")

            # Remove background - Set pixels further than clipping_distance to grey

            tic = time.time()
            depth_image_3d = np.dstack(
                (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
            toc = time.time()
            print(f"Time for stacking: {toc - tic:0.4f} seconds")

            tic = time.time()
            segmented_image = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), self.grey_color, color_image)
            toc = time.time()
            print(f"Time for filtering: {toc - tic:0.4f} seconds")

            return segmented_image

    def get_images(self, frames):

        tic = time.time()
        frames = self.pipeline.wait_for_frames()
        toc = time.time()
        print(f"Time for waiting for next frame: {toc - tic:0.4f} seconds")

        tic = time.time()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        toc = time.time()
        print(f"Time for getting frames: {toc - tic:0.4f} seconds")

        if not depth_frame or not color_frame:
            return

        tic = time.time()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        toc = time.time()
        print(f"Overall time for array creation: {toc - tic:0.4f} seconds")

        if not depth_frame or not color_frame:
            return

        return color_image, depth_image


    def stop_pipeline(self):
        self.pipeline.stop()

# start_time_current = time.time()
# start_time_old = time.time()

# try:
#     while True:
# time_at_start = time.time()
#
# start_time_old = start_time_current
# start_time_current = time.time()
#
#
#
# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
# images = np.hstack((bg_removed, depth_colormap))
#
# fps = 1 / (0.000001 + start_time_current - start_time_old)
# stats = "Output FPS: {}".format(int(fps))
#
# cv2.rectangle(images, (0, 0), (300, 25), (255, 0, 0), cv2.FILLED)
# font = cv2.FONT_HERSHEY_DUPLEX
# cv2.putText(images, stats, (6, 19), font, 0.5, (255, 255, 255), 1)
#
# cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('Align Example', bg_removed)
# key = cv2.waitKey(1)

# finally:
