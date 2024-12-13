
from yolact_edge.yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from yolact_edge.yolact_edge.data import COLORS
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
import math
# from system import system

class process_information():
    def __init__(self, fake_lane, fake_lane_sample, exist_lane_data, is_save_all_data, y_sample):
        self.exist = np.zeros(4, dtype = float)
        
        self.mask_exist_keep = '0000'
        self.remove_lane_flag = False
        self.change_lane_flag = False
        self.change_lane_flag_keep_num = 0
        self.y_sample = y_sample
        self.lane_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        shape = (300, 600, 3)
        self.out_put_inform = np.full(shape, 255).astype(np.uint8)
        self.offset = np.zeros(4)
        self.fake_lane = fake_lane
        self.fake_lane_sample = fake_lane_sample
        self.exist_lane_data = exist_lane_data
        self.y_sample_index = self.exist_lane_data[1, :, 1]
        self.is_save_all_data = is_save_all_data
        self.change_lane_virtual = {}
        for i in range(4):
            self.change_lane_virtual[str(i)] = self.exist_lane_data[i]

    def generate_mask(self, dets_out, img, mask_alpha=0.45):
        # Output the mask result based on the drivable area detection result
        # and some other information initialization

        img_gpu = torch.Tensor(img).cuda()
        img_gpu = img_gpu / 255.0
        self.h, self.w, _ = img_gpu.shape
        t = postprocess(dets_out, self.w, self.h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = 0)
        torch.cuda.synchronize()

        masks = t[3][:5]
        classes, scores, boxes = [x[:5].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(5, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < 0:
                num_dets_to_consider = j
                break
        
        if num_dets_to_consider == 0:
            # No detections found so just output the original image
            return (img_gpu * 255).byte().cpu().numpy()
        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        masks = masks[:num_dets_to_consider, :, :, None]

        color = [(67, 54, 244)]
        colors = torch.Tensor(color).cuda().float() / 255.
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)
        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        self.mask = masks_color_summand[:,:,0].cpu().numpy()
        self.output_img = (img_gpu * 255).byte().cpu().numpy()
        self.get_all_lane = {}
        self.match_lane = {}
        self.virtual_lane_idx = []
        self.lane_available = np.zeros(4, np.int32)
        self.plot_lane_xidx = np.zeros(4, np.int32)

    def lane_exist(self, change_way):
        # Initialization of Lane Line Confidence :
        # If the virtual lane line is covered by the drivable area, the confidence value is 0.85. If not 0.75.
        # If no initialization is required :
        # If the drivable area covers the virtual lane lines, add 0.01 to the confidence value.

        if change_way or np.count_nonzero(self.exist == 0) == 4:
            self.offset = np.zeros(4)
            self.exist = np.zeros(4, np.float64)
            self.changelane_offset = np.zeros(2)
            for i, lane in enumerate(self.exist_lane_data):
                for x, y in lane:
                    if self.mask[y][x]:
                        self.exist[i] = 0.85   #1.0 
                        break
                    else:
                        self.exist[i] = 0.75   #0.7
        else:
            for i, lane in enumerate(self.exist_lane_data):
                count = 0
                for x, y in lane:
                    if self.mask[y][x]:
                        count += 1
                    if count >=3:
                        if self.exist[i] < 1.3:
                            self.exist[i] += 0.01
                            break
        # Used to judge whether the mask of the two frames before and after has changed.
        # Only when the virtual lane line is covered by more than three points of the drivable area will it be judged as '1'.
        # others '0'.
        self.mask_exist = ''              
        for i, lane in enumerate(self.exist_lane_data):
                count = 0
                for x, y in lane:
                    if self.mask[y][x]:
                        count+=1
                if count >=3:
                    self.mask_exist += '1'
                else:
                    self.mask_exist += '0'

    def change_lane(self):
        # Determine if self.mask_exist has changed.
        # For the calculation method please see section 3.4.4 of the paper.
        # And use the result to judge the change of the next frame.

        if self.mask_exist != '0000' and self.mask_exist_keep != '0000' and self.change_lane_flag :
            a = int(self.mask_exist_keep, 2)
            b = int(self.mask_exist, 2)
            diff = (bin(a ^ b)[2:]).zfill(4)
            if diff != '0000':
                diff_idxf = diff.find('1')
                diff_idxb = diff.rfind('1')
                if diff_idxf == diff_idxb:
                    if diff_idxf == 0 or diff_idxf == 3:
                        if self.mask_exist[diff_idxf] == '0':
                            self.exist[diff_idxf] = self.exist[diff_idxf] - 0.45
                            if self.exist[diff_idxf] <= 0.75:
                                self.exist[diff_idxf] = 0.75
                        if self.mask_exist[diff_idxf] == '1':
                            self.exist[diff_idxf] = self.exist[diff_idxf] + 0.5
                            if self.exist[diff_idxf] >= 1.3:
                                self.exist[diff_idxf] = 1.3
                else:
                    if self.mask_exist[0] == self.mask_exist[-1]:
                        if self.mask_exist[0] == '0':
                            self.exist[0] = 0.75
                            self.exist[-1] = 0.75

                        if self.mask_exist == '1':
                            self.exist[0] = 0.85
                            self.exist[-1] = 0.85
                    else:
                        if self.mask_exist[0] == '0':
                            self.exist = self.exist[:3]
                            self.exist = np.insert(self.exist, 0, 0.75)
                        if self.mask_exist[0] == '1':
                            self.exist = self.exist[1:]
                            self.exist = np.insert(self.exist, 3, 0.75)
        self.mask_exist_keep = self.mask_exist

    def cat_lane_available(self):
        # When the confidence value is greater than 0.8, it is considered that this lane line should exist.

        for i in range(4):
            if self.exist[i] > 0.8:
                self.lane_available[i] = 1

    def lane_state(self):
        # Use self.lane_available to determine the state of the vehicle
        # Since the vehicle is bound to drive in the lane, it is only necessary to judge the left and right sides

        left = self.lane_available[0]
        right = self.lane_available[-1]
        if left == 1 and right == 1:
            self.state = 'mid'
        elif left == 0 and right == 1:
            self.state = 'left'
        elif left == 1 and right == 0:
            self.state = 'right'
        elif left == 0 and right == 0:
            self.state = 'mid'
        else:
            self.state = 'unknow'

    def cat_romve_lane(self, lanes):
        # Determine if lanes need to be removed.

        self.lanes = lanes
        if len(lanes) > np.count_nonzero(self.lane_available==1):
            self.remove_lane_flag = True
            self.remove_lane_num = len(self.lanes) - np.count_nonzero(self.lane_available==1)
        else:
            self.remove_lane_flag = False
            self.remove_lane_num = 0

    def distance_cost(self, predictions):
        # Repeat predictions and targets to generate all combinations, use the average distance as the distance cost.
        
        targets = self.fake_lane_sample
        targets = targets[:, self.idx]

        num_priors = predictions.shape[0]
        num_targets = targets.shape[0]

        predictions = np.repeat(predictions, num_targets, axis=0)
        targets = np.tile(targets, (num_priors, 1))
        invalid_masks = (targets < 0)
        lengths = (~invalid_masks).sum(axis=1)
        distances = abs((targets - predictions))
        distances[invalid_masks] = 0.

        distances = distances.sum(axis=1) / (lengths + 1e-9)
        distances = np.array(distances).reshape(num_priors, num_targets)
        self.distances = distances

    def match_pred_virtual(self):
        # Match virtual and predicted lane lines based on distance cost.
        # First, remove the locations where the lane line points are not predicted.
        # Similarly, remove the point corresponding to the virtual lane line.

        pred = []
        ycoords = []
        for _, lane in enumerate(self.lanes):
            lane_sample = []
            ycoord = []
            for y_idx in self.y_sample_index:
                lane_sample.append(0)
                for x, y in lane:
                    if y_idx == y:
                        if not (x < 0 or y < 0 or x > self.w or y > self.h):
                            lane_sample[-1] = int(x)
                            ycoord.append(int(y_idx))
                        break

            pred.append(lane_sample)
            ycoords.append(ycoord)
        pred = np.array(pred)
        same_index = set(self.y_sample_index)

        for y_coord in ycoords:
            set_y = set(y_coord)
            same_index &= set_y

        same_index = list(same_index)
        same_index.sort(reverse = True)
        self.idx = []
        for i in same_index:
            id_ = np.where(self.y_sample_index == i)
            self.idx.append(id_[0][0])

        pred = pred[:, self.idx]

        self.distance_cost(pred)
        self.pred_ind, self.virtual_ind = linear_sum_assignment(self.distances)

        # When a lane change can be triggered, we preserve the inner lane lines as much as possible.
        # Unless the distance cost of this lane line is small enough.

        if self.change_lane_flag :
            if 1 not in self.virtual_ind :
                if 0 in self.virtual_ind :
                    idx_ = np.where(self.virtual_ind == 0)[0][0]
                    if self.distances[self.pred_ind[idx_]][self.virtual_ind[idx_]] > 100:
                        self.virtual_ind[self.virtual_ind == 0] = 1
            elif 2 not in self.virtual_ind :  
                if 3 in self.virtual_ind :
                    idx_ = np.where(self.virtual_ind == 3)[0][0]
                    if self.distances[self.pred_ind[idx_]][self.virtual_ind[idx_]] > 100:
                        self.virtual_ind[self.virtual_ind == 3] = 2

    def modify_exist_virtual(self):
        # For virtual lane lines that are not matched successfully, reduce the confidence value by 0.01.

        virtual_lane_idx = [0, 1, 2, 3]
        virtual_idx = set(virtual_lane_idx)^set(self.virtual_ind)
        for i in virtual_idx:
            if self.exist[i] > 0.3:
                self.exist[i] = self.exist[i] - 0.01

    def remove_lane_stage_1(self):
        # When the number of predicted lane lines is greater than the number of lane lines that can exist.
        # We keep the inner predicted lane lines as much as possible.

        if self.remove_lane_flag:
            for i in range(self.remove_lane_num):
                if 0 in self.virtual_ind :
                    try:
                        vir_idx = np.where(self.virtual_ind==1)[0][0]
                        self.pred_ind = np.delete(self.pred_ind, vir_idx)
                        self.virtual_ind = np.delete(self.virtual_ind, vir_idx)
                        continue
                    except:
                        vir_idx = np.where(self.virtual_ind==2)[0][0]
                        self.pred_ind = np.delete(self.pred_ind, vir_idx)
                        self.virtual_ind = np.delete(self.virtual_ind, vir_idx)
                        continue
                if 3 in self.virtual_ind :
                    try:
                        vir_idx = np.where(self.virtual_ind==2)[0][0]
                        self.pred_ind = np.delete(self.pred_ind, vir_idx)
                        self.virtual_ind = np.delete(self.virtual_ind, vir_idx)
                        continue
                    except:
                        vir_idx = np.where(self.virtual_ind==1)[0][0]
                        self.pred_ind = np.delete(self.pred_ind, vir_idx)
                        self.virtual_ind = np.delete(self.virtual_ind, vir_idx)
                        continue

    def virtual_lane_stage_1(self):
        # Add 0.01 to the corresponding confidence value according to the matching result.
        # record the index for which the match failed.

        self.wait_idx = []
        for i in self.pred_ind:
            pred_lane = self.lanes[i]
            color_idx = np.where(self.pred_ind==i)[0][0]
            if self.exist[self.virtual_ind[color_idx]] < 1.3:
                self.exist[self.virtual_ind[color_idx]] = self.exist[self.virtual_ind[color_idx]] + 0.01
            if self.lane_available[self.virtual_ind[color_idx]] == 1:
                self.match_lane[str(self.virtual_ind[color_idx])] = pred_lane
                if self.is_save_all_data:
                    self.get_all_lane[str(self.virtual_ind[color_idx])] = pred_lane
                self.change_lane_virtual[str(self.virtual_ind[color_idx])] = pred_lane
                self.lane_available[self.virtual_ind[color_idx]] = 0
            else:
                self.wait_idx.append(i)

    def virtual_lane_stage_2(self):
        # Determine whether the left and right sides of the reserved index are empty.

        space_null = np.where(self.lane_available==1)
        if len(space_null[0]) != 0 and self.wait_idx:
            for i in self.wait_idx:
                pred_idx = np.where(self.pred_ind==i)[0][0]
                virtual_idx = self.virtual_ind[pred_idx]

                if virtual_idx-1 in space_null[0]:
                    pred_lane = self.lanes[i]
                    self.match_lane[str(self.virtual_ind[pred_idx])] = pred_lane
                    if self.is_save_all_data:
                        self.get_all_lane[str(self.virtual_ind[pred_idx])] = pred_lane
                    self.change_lane_virtual[str(self.virtual_ind[pred_idx])] = pred_lane
                    self.lane_available[virtual_idx-1] = 0

                elif virtual_idx+1 in space_null[0]:
                    
                    pred_lane = self.lanes[i]
                    self.match_lane[str(self.virtual_ind[pred_idx])] = pred_lane
                    if self.is_save_all_data:
                        self.get_all_lane[str(self.virtual_ind[pred_idx])] = pred_lane
                    self.change_lane_virtual[str(self.virtual_ind[pred_idx])] = pred_lane
                    self.lane_available[virtual_idx+1] = 0

    def virtual_lane_stage_3_change_lane(self):
        # Record the virtual lane lines corresponding to the remaining existing lane lines.

        space_null = np.where(self.lane_available==1) 
        self.virtual_lane_idx = []
        if len(space_null[0]) != 0:
            for i in range(len(space_null[0])):
                idx = space_null[0][i]
                if str(idx) in self.change_lane_virtual.keys():
                    virtual_lane = self.change_lane_virtual[str(idx)]
                    self.virtual_lane_idx.append(idx)
                    if self.is_save_all_data:
                        self.get_all_lane[str(idx)] = virtual_lane
                    self.lane_available[idx] = 0

    def remove_lane_stage_2(self):
        # Determine if the virtual lane line is too close.

        y_sample = self.y_sample_index[2:5]

        if len(self.virtual_lane_idx) > 1 :
            remove_idx = []
            compare_x = []
            for i in self.virtual_lane_idx: 
                lane = self.change_lane_virtual[str(i)]
                # print(i, lane)
                x_coord = []
                for y in y_sample :
                    if y in lane[:, 1]:
                        append_idx = np.where(lane[:, 1] == y)
                        x_coord.append(int(lane[append_idx[0][0]][0]))
                    else:
                        x_coord.append(0)
                compare_x.append(x_coord)
            compare_x = np.array(compare_x)
            # print(compare_x, compare_x.shape)
            end_flag = True
            if compare_x.shape[0] > 1:
                for j in range(compare_x.shape[1]):
                    # print(match[:, i])
                    if end_flag and 0 not in compare_x[:,j]:
                        compare_x_split = compare_x[:,j]
                        for j in range(len(compare_x_split) -1):
                            if abs(compare_x_split[j] - compare_x_split[j+1]) < 150 :
                                remove_idx.append(j)
                        end_flag = False
                    if end_flag == False:
                        break
                for i in remove_idx:
                    self.virtual_lane_idx.remove(self.virtual_lane_idx[i])

        # Determine if the virtual lane line is too close to the predicted lane line.

        remove_idx = []
        for i in self.virtual_lane_idx:
            
            compare_x = []
            if str(i) in self.change_lane_virtual.keys():
                lane = self.change_lane_virtual[str(i)]
                # print(i, lane)
                x_coord = []
                for y in y_sample :
                    if y in lane[:, 1]:
                        append_idx = np.where(lane[:, 1] == y)
                        x_coord.append(int(lane[append_idx[0][0]][0]))
                    else:
                        x_coord.append(0)
                compare_x.append(x_coord)
                # print('v1 c', compare_x)
            if str(i-1) in self.match_lane.keys():
                lane = self.match_lane[str(i-1)]
                # print(i, lane)
                x_coord = []
                for y in y_sample :
                    if y in lane[:, 1]:
                        append_idx = np.where(lane[:, 1] == y)
                        x_coord.append(int(lane[append_idx[0][0]][0]))
                    else:
                        x_coord.append(0)
                compare_x.insert(0, x_coord)
                # print('v2 c', compare_x)
            if str(i+1) in self.match_lane.keys():
                lane = self.match_lane[str(i+1)]
                # print(i, lane)
                x_coord = []
                for y in y_sample :
                    if y in lane[:, 1]:
                        append_idx = np.where(lane[:, 1] == y)
                        x_coord.append(int(lane[append_idx[0][0]][0]))
                    else:
                        x_coord.append(0)
                compare_x.append(x_coord)
                # print('v3 c', compare_x)

            compare_x = np.array(compare_x)
            # print(compare_x, compare_x.shape)
            end_flag = True
            if compare_x.shape[0] > 1:
                for j in range(compare_x.shape[1]):
                    # print(match[:, i])
                    # print( compare_x[:,j])
                    if end_flag and 0 not in compare_x[:,j]:
                        compare_x_split = compare_x[:,j]
                        # print(compare_x_split)
                        for j in range(len(compare_x_split) -1):
                            if abs(compare_x_split[j] - compare_x_split[j+1]) < 150 :
                                remove_idx.append(i)
                                end_flag = False
                                break
                    if end_flag == False:
                        break
        # print('remove stage 2 idx ', remove_idx, self.virtual_lane_idx)

        for i in remove_idx:
            self.virtual_lane_idx.remove(i)
        # print('after remove', self.virtual_lane_idx)

    def offset_compute(self):
        # Calculate offset based on matching result
        # Offset is not calculated if the number of predicted lane line points is less than three.

        target, match_lane = self.fake_lane_sample, self.match_lane
        offset_keep = self.offset.copy()
        if match_lane:
            match = []
            lane_keep_index = []
            for i in range(4):
                if str(i) in match_lane.keys():
                    match_data = []
                    lane = match_lane[str(i)]
                    # print(lane, self.y_sample)
                    for y in self.y_sample_index :
                        if y in lane[:, 1]:
                            append_idx = np.where(lane[:, 1] == y)
                            match_data.append(int(lane[append_idx[0][0]][0]))
                        else:
                            match_data.append(0)
                    match.append(match_data)
                    lane_keep_index.append(i)
            target = target[lane_keep_index, :]
            match = np.array(match)
            point_keep_index = []
            for i in range(match.shape[1]):
                if 0 not in match[:,i]:
                    point_keep_index.append(i)
            if len(point_keep_index) > 3:
                target = target[:, point_keep_index]
                match = match[:, point_keep_index]
                offset_cat = (target - match)
                offset_cat = offset_cat.sum(axis=1) / (len(point_keep_index) + 1e-9)
                idx = 0
                for i in lane_keep_index:
                    self.offset[i] = offset_cat[idx]
                    idx += 1
            else:
                print('offset point less then 4!!')
        self.offset = list(map(int, self.offset))

        # Calculate if the offset between two frames is greater than 10 pixels
        # If so, lane changing conditions can occur for the next five frames.

        change_lane_count = 0
        lane_num = 0
        for i in range(1, 3):
            if abs(offset_keep[i] - self.offset[i]) == 0:
                continue
            else:
                change_lane_count += abs(offset_keep[i] - self.offset[i])
                lane_num += 1
        if lane_num >= 1:
            change_lane_count /= lane_num
            if change_lane_count >= 10 :
                self.change_lane_flag = True
                self.change_lane_flag_keep_num = 0
                
            else:
                if self.change_lane_flag == True :
                    self.change_lane_flag_keep_num += 1
                    if self.change_lane_flag_keep_num > 5:
                        self.change_lane_flag = False
                else:
                    self.change_lane_flag = False
        else:
            self.change_lane_flag = False

    def draw_result(self):
        # Plot the result on the image

        exist = []
        for i in self.exist:
            exist.append(round(i,2))
        self.out_put_inform = cv2.putText(self.out_put_inform, 'exist : '+ str(exist), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        self.out_put_inform = cv2.putText(self.out_put_inform, 'offset : '+ str(self.offset), (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        self.out_put_inform = cv2.putText(self.out_put_inform, 'state : ' + self.state, (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)    
        
        for i in self.match_lane.keys():
            lane = self.match_lane[str(i)]
            keep_x, keep_y = 0, 0
            for x, y in lane:
                if x <= 0 or y <= 0 or x > self.w or y > self.h:
                    continue
                x, y = int(x), int(y)
                if keep_x  == 0 and keep_y == 0:
                    keep_x = x
                    keep_y = y
                    continue
                if keep_x != x and keep_y != y:
                    self.output_img = cv2.line(self.output_img, (keep_x, keep_y), (x, y), self.lane_color[int(i)], 4)
                    keep_x = x
                    keep_y = y

        for i in self.virtual_lane_idx:
            lane = self.change_lane_virtual[str(i)]              
            keep_x, keep_y = 0, 0
            for x, y in lane:
                if x <= 0 or y <= 0 or x > self.w or y > self.h:
                    continue
                x, y = int(x), int(y)
                if keep_x  == 0 and keep_y == 0:
                    keep_x = x
                    keep_y = y
                    continue
                if keep_x != x and keep_y != y:
                    self.output_img = cv2.line(self.output_img, (keep_x, keep_y), (x, y), (255, 255, 255), 4)
                    keep_x = x
                    keep_y = y

        for lane in self.exist_lane_data:
            for i in range(len(lane)):
                self.output_img = cv2.circle(self.output_img, lane[i], 1, (0, 0, 0), 2)
        self.output_img_noinfo = self.output_img.copy()
        self.output_img[:300, :600] = self.out_put_inform
        shape = (300, 600, 3)
        self.out_put_inform = np.full(shape, 255).astype(np.uint8)

class process_gps():
    def __init__(self):
        self.offset_keep = 0
        self.lat = 110936.2
        self.lon = 101775.45
    def offset_shift(self):
        # Calculate the offset distance
        # It is preset to 2 lane lines here,
        # and the condition of 3 lane lines is only on the highway scenes with different types of dash cam.
        #　So in the future three-lane conditions it is necessary to readjust the calculation of the offset distance.

        if self.lane_num != 0:
            if self.lane_num >= 3:
                self.shift_width = 3.5
            else:
                self.shift_width = 1.75
        else:
            self.shift_width = 1.75

        if self.lane_num >= 3:
            if 30 < abs(self.offset_keep) <= 60 :       # 20 60 
                shift = 0.25 * (self.shift_width/1.75)
            elif 60 < abs(self.offset_keep) <= 100:
                shift = 0.5 * (self.shift_width/1.75)
            elif 100 < abs(self.offset_keep):
                shift = 0.875 * (self.shift_width/1.75)
            else:
                shift = 0

        else:
            if 30 < abs(self.offset_keep) <= 80 :       # 20 60 
                shift = 0.5 * (self.shift_width/1.75)
            elif 80 < abs(self.offset_keep) <= 150:
                shift = 1.0 * (self.shift_width/1.75)
            elif 150 < abs(self.offset_keep):
                shift = 1.75 * (self.shift_width/1.75)
            else:
                shift = 0

        if self.state == 'left':
            self.default_lane_width = -self.shift_width
            if self.offset_keep > 0:
                self.offset_width = self.default_lane_width + shift
            else:
                self.offset_width = self.default_lane_width - shift
        elif self.state =='right':
            self.default_lane_width = self.shift_width
            if self.offset_keep > 0:
                self.offset_width = self.default_lane_width + shift
            else:
                self.offset_width = self.default_lane_width - shift
        else:
            self.default_lane_width = 0
            if self.offset_keep > 0:
                self.offset_width = self.default_lane_width + shift
            else:
                self.offset_width = self.default_lane_width - shift

    def gps_shift(self):
        # Shift GPS based on offset distance.

        vector = [(self.node_2_coord[0] - self.node_1_coord[0]) * self.lat, (self.node_2_coord[1] - self.node_1_coord[1]) * self.lon]

        norm = [vector[1]*-1,vector[0]]
        
        unit_length = math.sqrt(norm[0]*norm[0]+norm[1]*norm[1])
        unitNorm = [norm[0]/unit_length,norm[1]/unit_length]
        dot = unitNorm[0]*vector[0]+unitNorm[1]*vector[1]

        new_x = self.project[0] + unitNorm[0]*(self.offset_width)/self.lat
        new_y = self.project[1] + unitNorm[1]*(self.offset_width)/self.lon
        
        self.project = [new_x, new_y]
        

    def cat_offset(self):
        # Calculate the offset size based on the two predicted lane lines in the middle.

        offset_left = self.offset_record[1]
        offset_right = self.offset_record[2]
        if offset_left != 0 and offset_right != 0:
            self.offset_keep = (offset_right + offset_left)/2
        self.offset_shift()
        self.gps_shift()


    def project_modify(self, project, offset_record, state, node_1_coord = None, node_2_coord = None, lane_num = None):
        # Calculate offset and shift GPS based on number of lanes.

        self.offset_record = offset_record
        self.state = state
        self.node_1_coord = node_1_coord
        self.node_2_coord = node_2_coord
        self.lane_num = lane_num
        self.project = [project[0], project[1]]
        if self.lane_num < 3:
            if state == 'left' or state =='right':
                self.cat_offset()
            else:
                print('state = mid')
        else:
            self.cat_offset()
            
        
