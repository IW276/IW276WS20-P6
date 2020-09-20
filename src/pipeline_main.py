import itertools
import time
import cv2
import face_recognition
import numpy as np
import json
from queue import Queue 
from threading import Thread 
from face_expression_recognition import TRTModel
from realsense_frame_service import RealsenseFrameService
from text_export import TextExport

class CurrentIterationItem:

    time_after_expr_rec = 0
    color_frame = None
    depth_frame = None
    segmented_frame = None
    face_locations = []
    face_expressions = []

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

    def get_next_frame(self, current_iteration_item):

        tic = time.time()
        color_frame, depth_frame, segmented_frame = self.realsense_frame_service.fetch_images(current_iteration_item.process_next_frame)
        toc = time.time()
        print(f"Overall time for segmentation: {toc - tic:0.4f} seconds")
        return color_frame, depth_frame, segmented_frame

    def process_frame(self, current_iteration_item):

        # face recognition
        if current_iteration_item.process_next_frame:
            segmented_frame = current_iteration_item.segmented_frame
            small_frame = cv2.resize(
                segmented_frame, (0, 0), fx=1 / self.scale_factor, fy=1 / self.scale_factor)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            current_iteration_item.face_locations = face_locations
            time_after_face_rec = time.time()
            print("Time Face Recognition: {:.2f}".format(
                time_after_face_rec - current_iteration_item.time_at_start))

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
                
            current_iteration_item.face_expressions = face_expressions
            time_after_expr_rec = time.time()
            current_iteration_item.time_after_expr_rec = time_after_expr_rec
            if len(face_expressions) > 0:
                print("Time Face Expression Recognition: {:.2f}".format(
                    time_after_expr_rec - time_after_face_rec))
        else:
            cv2.waitKey(33)

        return current_iteration_item

    def generate_output(self, _cv2, current_iteration_item):

    # graphical output face expression recognition
        color_frame = current_iteration_item.color_frame

        for (top, right, bottom, left), face_expression in itertools.zip_longest(current_iteration_item.face_locations, 
                                                                                current_iteration_item.face_expressions,
                                                                                fillvalue=''):
            top *= self.scale_factor
            right *= self.scale_factor
            bottom *= self.scale_factor
            left *= self.scale_factor
            _cv2.rectangle(color_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            _cv2.rectangle(color_frame, (left, bottom),
                        (right, bottom + 25), (0, 0, 255), _cv2.FILLED)
            font = _cv2.FONT_HERSHEY_DUPLEX
            _cv2.putText(current_iteration_item.color_frame, face_expression, (left + 6, bottom + 18),
                        font, 0.8, (255, 255, 255), 1)

        # graphical output stats
        fps = self.fps_constant / (current_iteration_item.start_time_current - current_iteration_item.start_time_old)
        stats = "Output FPS: {} | Frame: {}".format(int(fps), current_iteration_item.frame_number)
        _cv2.rectangle(color_frame, (0, 0), (300, 25), (255, 0, 0), _cv2.FILLED)
        font = _cv2.FONT_HERSHEY_DUPLEX
        _cv2.putText(color_frame, stats, (6, 19), font, 0.5, (255, 255, 255), 1)
        print("Output formatting: {:.2f}".format(time.time() - current_iteration_item.time_after_expr_rec))

        # display resulting image
        depth_colormap = _cv2.applyColorMap(_cv2.convertScaleAbs(current_iteration_item.depth_frame, alpha=0.03), _cv2.COLORMAP_JET)
        _cv2.namedWindow('Video', _cv2.WINDOW_AUTOSIZE)
        return np.hstack((color_frame, depth_colormap)), _cv2

    def append_to_output_json(self, current_iteration_item):
        # log when 'l' is being pressed
        # if cv2.waitKey(1) & 0xFF == ord('l'):
        for (top, right, bottom, left), face_expression in itertools.zip_longest(current_iteration_item.face_locations, 
                                                                                current_iteration_item.face_expressions, 
                                                                                fillvalue=''):      

            self.export.append(current_iteration_item.frame_number, (top, left), (right, bottom), face_expression)

    def json_output_loop(self, process_frame_queue):
        
        while True:
            current_iteration_item = process_frame_queue.get()
            self.append_to_output_json(current_iteration_item)

    def video_output_loop(self, process_frame_queue):

        while True:

            current_iteration_item = process_frame_queue.get()
            double_img, _cv2 = self.generate_output(cv2, current_iteration_item)
            _cv2.imshow('Video', double_img)

    def process_frame_loop(self, next_frame_queue, process_frame_queue):

        while True:

            current_iteration_item = next_frame_queue.get()
            current_iteration_item = self.process_frame(current_iteration_item)
            process_frame_queue.put(current_iteration_item)

    def next_frame_loop(self, next_frame_queue):

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
            current_iteration_item = CurrentIterationItem(start_time_current, start_time_old, time_at_start, process_next_frame, frame_number)
            color_frame, depth_frame, segmented_frame = self.get_next_frame(current_iteration_item)
            current_iteration_item.color_frame = color_frame
            current_iteration_item.depth_frame = depth_frame
            current_iteration_item.segmented_frame = segmented_frame
            next_frame_queue.put(current_iteration_item)

            frame_number += 1

    def processing_loop(self):

        next_frame_queue = Queue() 
        process_frame_queue = Queue()
        next_frame_thread = Thread(target = self.next_frame_loop, args =(next_frame_queue,)) 
        process_frame_thread = Thread(target = self.process_frame_loop, args =(next_frame_queue, process_frame_queue,))
        json_output_thread = Thread(target = self.json_output_loop, args =(process_frame_queue,))
   
        next_frame_thread.start() 
        process_frame_thread.start() 
        json_output_thread.start() 
        self.video_output_loop(process_frame_queue)

        # break when 'q' is being pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.export.close()
            # break

        self.export.close()
        cv2.destroyAllWindows()
    
pipeline = Pipeline()
pipeline.processing_loop()


