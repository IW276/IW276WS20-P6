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

while True:
    time_at_start = time.time()
    print("Frame: {}".format(frame_number))
    # set timers for FPS calculation
    if frame_number % fps_constant == 0:
        start_time_old = start_time_current
        start_time_current = time.time()

    processNextFrame = frame_number % process_Nth_frame == 0

    tic = time.time()
    color_image, depth_image, segmented_image = realsense_frame_service.fetch_images(processNextFrame)
    frame = color_image
    toc = time.time()
    print(f"Overall time for segmentation: {toc - tic:0.4f} seconds")

    # face recognition
    if processNextFrame:
        small_frame = cv2.resize(
            frame, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        time_after_face_rec = time.time()
        print("Time Face Recognition: {:.2f}".format(
            time_after_face_rec - time_at_start))

        # face expression recognition
        face_expressions = []
        for (top, right, bottom, left) in face_locations:
            # Magic Face Expression Recognition
            face_image = frame[top * scale_factor:bottom *
                                                  scale_factor, left * scale_factor:right * scale_factor]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_exp = face_exp_rec.face_expression(face_image)
            face_expressions.append(face_exp)

        time_after_expr_rec = time.time()
        if len(face_expressions) > 0:
            print("Time Face Expression Recognition: {:.2f}".format(
                time_after_expr_rec - time_after_face_rec))
    # else:
        # cv2.waitKey(33)
        

    frame_number += 1

    def generateOutput(open_cv_2):

        # graphical output face expression recognition
        for (top, right, bottom, left), face_expression in itertools.zip_longest(face_locations, face_expressions,
                                                                                fillvalue=''):
            top *= scale_factor
            right *= scale_factor
            bottom *= scale_factor
            left *= scale_factor
            open_cv_2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            open_cv_2.rectangle(frame, (left, bottom),
                        (right, bottom + 25), (0, 0, 255), open_cv_2.FILLED)
            font = open_cv_2.FONT_HERSHEY_DUPLEX
            open_cv_2.putText(frame, face_expression, (left + 6, bottom + 18),
                        font, 0.8, (255, 255, 255), 1)

        # graphical output stats
        fps = fps_constant / (start_time_current - start_time_old)
        stats = "Output FPS: {} | Frame: {}".format(int(fps), frame_number)
        open_cv_2.rectangle(frame, (0, 0), (300, 25), (255, 0, 0), open_cv_2.FILLED)
        font = open_cv_2.FONT_HERSHEY_DUPLEX
        open_cv_2.putText(frame, stats, (6, 19), font, 0.5, (255, 255, 255), 1)
        print("Output formatting: {:.2f}".format(
            time.time() - time_after_expr_rec))

        # display resulting image
        depth_colormap = open_cv_2.applyColorMap(open_cv_2.convertScaleAbs(depth_image, alpha=0.03), open_cv_2.COLORMAP_JET)
        open_cv_2.namedWindow('Video', open_cv_2.WINDOW_AUTOSIZE)
        return np.hstack((frame, depth_colormap)), open_cv_2

    with concurrent.futures.ThreadPoolExecutor() as executor:

        future_output = executor.submit(generateOutput, cv2)

        double_img, open_cv_2 = future_output.result()
        open_cv_2.imshow(double_img)
        
    
    # log when 'l' is being pressed
    # if cv2.waitKey(1) & 0xFF == ord('l'):

    def appendToOutput():
        for (top, right, bottom, left), face_expression in itertools.zip_longest(face_locations, face_expressions, fillvalue=''):                                                   
            export.append(frame_number, (top, left), (right, bottom), face_expression)

    # # break when 'q' is being pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     export.close()
    #     break

export.close()
cv2.destroyAllWindows()
