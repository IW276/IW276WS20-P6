import itertools
import time
import cv2
import face_recognition
import numpy as np
import json
import concurrent.futures
from face_expression_recognition import TRTModel
from realsense_frame_service import RealsenseFrameService
from text_export import TextExport

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
frame_number = 0
face_locations = []
face_expressions = []
cropped = 0
start_time_current = time.time()
start_time_old = time.time()

def get_next_frame(process_next_frame):

    tic = time.time()
    color_frame, depth_frame, segmented_frame = realsense_frame_service.fetch_images(process_next_frame)
    toc = time.time()
    print(f"Overall time for segmentation: {toc - tic:0.4f} seconds")
    return color_frame, depth_frame, segmented_frame

def process_frame(color_frame, depth_frame, segmented_frame):

    # face recognition
    if process_next_frame:
        small_frame = cv2.resize(
            segmented_frame, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        time_after_face_rec = time.time()
        print("Time Face Recognition: {:.2f}".format(
            time_after_face_rec - time_at_start))

        # face expression recognition
        face_expressions = []
        for (top, right, bottom, left) in face_locations:
            # Magic Face Expression Recognition
            face_image = segmented_frame[top * scale_factor:bottom *
                                                scale_factor, left * scale_factor:right * scale_factor]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_exp = face_exp_rec.face_expression(face_image)
            face_expressions.append(face_exp)

        time_after_expr_rec = time.time()
        if len(face_expressions) > 0:
            print("Time Face Expression Recognition: {:.2f}".format(
                time_after_expr_rec - time_after_face_rec))
    
    return color_frame, depth_frame

def generate_output(_cv2, color_frame, depth_frame):

# graphical output face expression recognition
    for (top, right, bottom, left), face_expression in itertools.zip_longest(face_locations, face_expressions,
                                                                            fillvalue=''):
        top *= scale_factor
        right *= scale_factor
        bottom *= scale_factor
        left *= scale_factor
        _cv2.rectangle(color_frame, (left, top), (right, bottom), (0, 0, 255), 2)
        _cv2.rectangle(color_frame, (left, bottom),
                    (right, bottom + 25), (0, 0, 255), _cv2.FILLED)
        font = _cv2.FONT_HERSHEY_DUPLEX
        _cv2.putText(color_frame, face_expression, (left + 6, bottom + 18),
                    font, 0.8, (255, 255, 255), 1)

    # graphical output stats
    fps = fps_constant / (start_time_current - start_time_old)
    stats = "Output FPS: {} | Frame: {}".format(int(fps), frame_number)
    _cv2.rectangle(color_frame, (0, 0), (300, 25), (255, 0, 0), _cv2.FILLED)
    font = _cv2.FONT_HERSHEY_DUPLEX
    _cv2.putText(color_frame, stats, (6, 19), font, 0.5, (255, 255, 255), 1)
    # print("Output formatting: {:.2f}".format(time.time() - time_after_expr_rec))

    # display resulting image
    depth_colormap = _cv2.applyColorMap(_cv2.convertScaleAbs(depth_frame, alpha=0.03), _cv2.COLORMAP_JET)
    _cv2.namedWindow('Video', _cv2.WINDOW_AUTOSIZE)
    return np.hstack((color_frame, depth_colormap)), _cv2

def append_to_output_json():
    # log when 'l' is being pressed
    # if cv2.waitKey(1) & 0xFF == ord('l'):
    for (top, right, bottom, left), face_expression in itertools.zip_longest(face_locations, face_expressions, fillvalue=''):                                                   
        export.append(frame_number, (top, left), (right, bottom), face_expression)

def video_output_callback(video_output_future):
    double_img, _cv2 = video_output_future.result()
    _cv2.imshow('Video', double_img)

def process_frame_callback(process_frame_future, executor):
    color_frame, depth_frame = process_frame_future.result()
    video_output_future = executor.submit(generate_output, cv2, color_frame, depth_frame)
    json_output_future = executor.submit(append_to_output_json)
    video_output_future.add_done_callback(video_output_callback)

def next_frame_callback(next_frame_future, executor):
    color_frame, depth_frame, segmented_frame = next_frame_future.result()
    process_frame_future = executor.submit(process_frame, color_frame, depth_frame, segmented_frame)
    process_frame_future.add_done_callback(process_frame_callback, executor)

with concurrent.futures.ThreadPoolExecutor() as executor:

    while True:

        time_at_start = time.time()
        print("Frame: {}".format(frame_number))
        # set timers for FPS calculation
        if frame_number % fps_constant == 0:
            start_time_old = start_time_current
            start_time_current = time.time()
        
        process_next_frame = frame_number % process_Nth_frame == 0

        next_frame_future = executor.submit(get_next_frame)
        next_frame_future.add_done_callback(next_frame_callback)

        frame_number += 1

        # break when 'q' is being pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            export.close()
            break

export.close()
cv2.destroyAllWindows()
