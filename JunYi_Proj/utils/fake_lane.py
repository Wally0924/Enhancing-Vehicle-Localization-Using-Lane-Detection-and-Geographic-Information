import numpy as np
import cv2

def create_fake_lane(fake_lane_txt, y_sample):
    # Read information from recorded virtual lane lines.
    # Virtual lane lines for different videos are in fake_lane's folder.

    f= open(fake_lane_txt, 'r')
    fake_lane = {}
    fake_lane_sample = []
    exist_lane_data = []
    for line in f.readlines():
        lane_idx, lane = line.strip().split(',')
        lane_data = []
        lane = lane.split(' ')
        a, b = lane[:2]
        fake_lane[lane_idx] = [float(a), float(b)]
        lane = list(map(int, lane[2:]))
        exist_lane = []
        max_number_of_lane = 20
        for y in y_sample:
            if y in lane:
                lane_data.append(lane[lane.index(y)-1])
                exist_lane.append((lane[lane.index(y)-1] , y))
        if max_number_of_lane > len(lane_data):
            max_number_of_lane = len(lane_data)
        fake_lane_sample.append(lane_data)
        exist_lane_data.append(exist_lane)
    f.close()
    
    for i in range(len(fake_lane_sample)):
        index_len = len(fake_lane_sample[i]) - max_number_of_lane 
        fake_lane_sample[i] = fake_lane_sample[i][index_len:]
    fake_lane_sample = np.array(fake_lane_sample)

    for i in range(len(exist_lane_data)):
        index_len = len(exist_lane_data[i]) - max_number_of_lane 
        exist_lane_data[i] = exist_lane_data[i][index_len:]
    exist_lane_data = np.array(exist_lane_data)

    return fake_lane, fake_lane_sample, exist_lane_data