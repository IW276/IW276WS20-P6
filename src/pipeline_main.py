import itertools
import time
import cv2
import face_recognition
import numpy as np
import json
import asyncio
from face_expression_recognition import TRTModel
from realsense_frame_service import RealsenseFrameService
from text_export import TextExport

class CurrentIterationItems:

    time_after_face_rec = 0

    def __init__(self, start_time_current, start_time_old, time_at_start, process_next_frame, frame_number):
        self.start_time_current = start_time_current
        self.start_time_old = start_time_old
        self.time_at_start = time_at_start
        self.process_next_frame = process_next_frame
        self.frame_number = frame_number

class Pipeline():

    with open("config.json", "r") as json_config_file:
        config_properties = json.load(json_config_file)

    # global variables
    fps_constant = int(config_properties["fpsConstant"])
    process_Nth_frame = int(config_properties["processNthFrame"])
    scale_factor = int(config_properties["scaleFactor"])
    target_width = int(config_properties["targetWidth"])
    resize_input = config_properties["useTargetSize"]

    # initialize face expression recognition
    print("Initializing Model...")
    face_exp_rec = TRTModel()
    realsense_frame_service = RealsenseFrameService()
    print("Done.")

    # init some variables
    export = TextExport()
    face_locations = []
    face_expressions = []
    cropped = 0

    async def get_next_frame(self, current_iteration_items):

        tic = time.time()
        color_frame, depth_frame, segmented_frame = self.realsense_frame_service.fetch_images(current_iteration_items.process_next_frame)
        toc = time.time()
        print(f"Overall time for segmentation: {toc - tic:0.4f} seconds")
        return color_frame, depth_frame, segmented_frame

    async def process_frame(self, segmented_frame, current_iteration_items):

        # face recognition
        if current_iteration_items.process_next_frame:
            small_frame = cv2.resize(
                segmented_frame, (0, 0), fx=1 / self.scale_factor, fy=1 / self.scale_factor)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            time_after_face_rec = time.time()
            current_iteration_items.time_after_face_rec = time_after_face_rec
            print("Time Face Recognition: {:.2f}".format(
                time_after_face_rec - current_iteration_items.time_at_start))

            # face expression recognition
            face_expressions = []
            for (top, right, bottom, left) in face_locations:
                # Magic Face Expression Recognition
                face_image = segmented_frame[top * self.scale_factor:bottom 
                                                * self.scale_factor, left 
                                                * self.scale_factor:right 
                                                * self.scale_factor]

                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_exp = self.face_exp_rec.face_expression(face_image)
                face_expressions.append(face_exp)

            time_after_expr_rec = time.time()
            if len(face_expressions) > 0:
                print("Time Face Expression Recognition: {:.2f}".format(
                    time_after_expr_rec - time_after_face_rec))

    async def generate_output(self, _cv2, color_frame, depth_frame, current_iteration_items):

    # graphical output face expression recognition
        for (top, right, bottom, left), face_expression in itertools.zip_longest(self.face_locations, self.face_expressions,
                                                                                fillvalue=''):
            top *= self.scale_factor
            right *= self.scale_factor
            bottom *= self.scale_factor
            left *= self.scale_factor
            _cv2.rectangle(color_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            _cv2.rectangle(color_frame, (left, bottom),
                        (right, bottom + 25), (0, 0, 255), _cv2.FILLED)
            font = _cv2.FONT_HERSHEY_DUPLEX
            _cv2.putText(color_frame, face_expression, (left + 6, bottom + 18),
                        font, 0.8, (255, 255, 255), 1)

        # graphical output stats
        fps = self.fps_constant / (current_iteration_items.start_time_current - current_iteration_items.start_time_old)
        stats = "Output FPS: {} | Frame: {}".format(int(fps), current_iteration_items.frame_number)
        _cv2.rectangle(color_frame, (0, 0), (300, 25), (255, 0, 0), _cv2.FILLED)
        font = _cv2.FONT_HERSHEY_DUPLEX
        _cv2.putText(color_frame, stats, (6, 19), font, 0.5, (255, 255, 255), 1)
        print("Output formatting: {:.2f}".format(time.time() - current_iteration_items.time_after_expr_rec))

        # display resulting image
        depth_colormap = _cv2.applyColorMap(_cv2.convertScaleAbs(depth_frame, alpha=0.03), _cv2.COLORMAP_JET)
        _cv2.namedWindow('Video', _cv2.WINDOW_AUTOSIZE)
        return np.hstack((color_frame, depth_colormap)), _cv2

    async def append_to_output_json(self, current_iteration_items):
        # log when 'l' is being pressed
        # if cv2.waitKey(1) & 0xFF == ord('l'):
        for (top, right, bottom, left), face_expression in itertools.zip_longest(self.face_locations, self.face_expressions, fillvalue=''):                                                   
            self.export.append(current_iteration_items.frame_number, (top, left), (right, bottom), face_expression)

    async def async_video_output(self, color_frame, depth_frame, current_iteration_items):
        self.append_to_output_json(current_iteration_items)
        double_img, _cv2 = await self.generate_output(cv2, color_frame, depth_frame, current_iteration_items)
        _cv2.imshow('Video', double_img)

    async def async_process_frame(self, color_frame, depth_frame, segmented_frame, current_iteration_items):
        await self.process_frame(segmented_frame, current_iteration_items)
        self.async_video_output(color_frame, depth_frame, current_iteration_items)

    async def async_next_frame(self, current_iteration_items):
        color_frame, depth_frame, segmented_frame = await self.get_next_frame(current_iteration_items)
        self.async_process_frame(color_frame, depth_frame, segmented_frame, current_iteration_items)

    async def processing_loop(self):
        
        frame_number = 0
        start_time_current = time.time()
        start_time_old = time.time()

        while True:

            time_at_start = time.time()
            print("Frame: {}".format(frame_number))
            # set timers for FPS calculation
            if frame_number % self.fps_constant == 0:
                start_time_old = start_time_current
                start_time_current = time.time()
            
            process_next_frame = frame_number % self.process_Nth_frame == 0
            current_iteration_items = CurrentIterationItems(start_time_current, start_time_old, time_at_start, process_next_frame, frame_number)

            await self.async_next_frame(current_iteration_items)

            frame_number += 1

            # break when 'q' is being pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.export.close()
                break

        self.export.close()
        cv2.destroyAllWindows()

async def main(): 
    
    pipeline = Pipeline()
    await pipeline.processing_loop()

asyncio.run(main())