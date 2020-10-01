# this class contains all relevant infomation 
# which will be propagated through the sections of the pipeline.
# each pipeline function takes one object as parameter and reads, writes 
# and/or returns the modified object for following operations.
class CurrentIterationItem:

    time_after_expr_rec = 0
    color_frame = None
    depth_frame = None
    segmented_frame = None
    _cv2 = None

    def __init__(self, start_time_current, start_time_old, time_at_start, process_next_frame, frame_number):
        self.start_time_current = start_time_current
        self.start_time_old = start_time_old
        self.time_at_start = time_at_start
        self.process_next_frame = process_next_frame
        self.frame_number = frame_number