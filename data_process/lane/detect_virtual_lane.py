import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
import math
from generate_lane_data import LabelFile
import time
from torchvision import transforms
from PIL import Image
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]

def hough(accumulator, diag_len, num_thetas, cos_t, sin_t, lanes = None):
    ## 220531 
    lane_dict = {}
    for lane in lanes:
        for x, y in lane:
            # print('x = ', x, y)
            if y == 860:
                
                if x < 384:
                    lane_dict['0'] = lane
                    break
                if 384 <= x < 960:
                    lane_dict['1'] = lane
                    break
                if 960 <= x < 1536:
                    lane_dict['2'] = lane
                    break
                if 1536 <= x < 1920:
                    lane_dict['3'] = lane
                    break
    # print(lane_dict.keys())
    ## 160720
    # lane_dict = {}
    # for lane in lanes:
    #     for x, y in lane:
    #         # print(x, y)
    #         if y == 560:
    #             if x < 250:
    #                 lane_dict['0'] = lane
    #                 break
    #             if 250 < x < 620:
    #                 lane_dict['1'] = lane
    #                 break
    #             if 620 < x < 1030:
    #                 lane_dict['2'] = lane
    #                 break
    #             if 1030 < x < 1280:
    #                 lane_dict['3'] = lane
    #                 break
    #     print('------------')
    # print(lane_dict.keys())           
    for key in lane_dict.keys():
        lane = lane_dict[key]
        # print(z)
        # input()
        for x, y in lane:
            for t_idx in range(num_thetas):
                rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
                accumulator[int(key), rho, t_idx] += 0.01
    return accumulator



class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        print(self.processes)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        self.y_sample = np.arange(980, 600, -40)# (880, 600, -40)   #(780, 420, -20)
        self.transform= transforms.Compose([
                        transforms.Resize([300, 800]),
                        transforms.ToTensor()
                        ])
        load_network(self.net, './best_new.pth')
        device = torch.device('cuda:0')

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.heads.get_lanes(data)
        # print(data)
        return data
        
    def hough(self, ori_img):

        img = ori_img[self.cfg.cut_height:, :, :]#.astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'ori_img':ori_img})

        data['lanes'] = self.inference(data)[0]
        # y_sample = np.arange(680, 1080, 20)
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        lane_xy = []
        for lane in lanes:
            xy = []
            for x, y in lane:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                # print('===', x, y, round(y, -1))
                if round(y, -1) in self.y_sample:
                    xy.append((x, round(y, -1)))
            lane_xy.append(xy)
        # print(lane_xy)
        for idx, xys in enumerate(lane_xy):
            # all_lane[str(idx)] = xys
            for i in range(1, len(xys)):
                data['ori_img'] = cv2.line(data['ori_img'], xys[i - 1], xys[i], COLORS[idx], thickness=4)
        data['ori_img'] =  cv2.resize(data['ori_img'], (640, 360))
        cv2.imshow('tt', data['ori_img'])
        cv2.waitKey(1)
        return lane_xy

def process_hough(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)

    thetas = np.deg2rad(np.arange(0.0, 180.0, 1))
    width, height = 1920, 1080   # 1920, 1080 1280, 800 
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    accumulator = np.zeros((4, 2 * diag_len, num_thetas), dtype=np.float)
    y_sample = np.arange(560, 1080, 20)
    # y_sample = np.arange(680, 1080, 20)  # 
    # y_sample = np.arange(440, 800, 20)
    pi = 3.1415926
    timeF = 2
    
    for file in os.listdir(args.img):
        print('eval video', file)
        cap = cv2.VideoCapture(args.img + file)
        # f = open('./.txt')
        rval, frame = cap.read()
        c = 1
        while rval:  
            
            
            if c % timeF == 0:
                # try:
                lane_xy = detect.hough(frame)

                accumulator = hough(accumulator, diag_len, num_thetas, cos_t, sin_t, lane_xy)
                img_show_lane = np.zeros((height, width), dtype=np.uint8)
                f_lane = open('./fake_lane.txt', 'w')
                for i in range(4):
                    accumulator_split = accumulator[i]
                    if i == 0:
                        accumulator_split[:,70:] = 0
                        # accumulator_split[:,:45] = 0
                    elif i == 1:
                        accumulator_split[:,80:] = 0
                        accumulator_split[:,:30] = 0
                    elif i == 2:
                        accumulator_split[:,:100] = 0
                        accumulator_split[:,145:] = 0
                    elif i == 3:
                        accumulator_split[:,:110] = 0
                        # accumulator_split[:,165:] = 0
                    r, th = np.where(accumulator_split==np.max(accumulator_split))
                    print(len(r))
                    # input()
                    if len(r) != 793080 :  # 793080 # 543240
                    # print(r, th)
                        print(c, i, len(th), len(r), np.max(accumulator_split))
                        for h in range(len(r)):
                            line_list = []
                            r_i = r[h] - diag_len
                            th_i = th[h]

                            x = r_i * math.cos(th_i * pi / 180)
                            y = r_i * math.sin(th_i * pi / 180)
                            a = math.tan((90+th_i)*pi/180)
                            b = y-a*x    
                            x = (y_sample-b)/a
                            x = list(map(int, x)) 
                            f_lane.write(str(i) + ',')
                            f_lane.write(str(a)+' '+str(b)+' ')
                            for j in range(len(y_sample)-1):
                                if 0 < x[j] < 1920 and 590 < y_sample[j] < 1080 :  # 1280 420, 800
                                    line_list.append((int(x[j]), int(y_sample[j])))
                                    f_lane.write(str(x[j])+' '+str(y_sample[j])+' ')
                            f_lane.write('\n')
                            
                            for k in range(len(line_list)-1):
                                cv2.line(img_show_lane,line_list[k], line_list[k+1], 255, 1)
                f_lane.close()
                img_show_lane = cv2.resize(img_show_lane, (640, 360))
                # cv2.imwrite('./fake_lane.png', img)
                cv2.imshow('ttt', img_show_lane)
                cv2.waitKey(1)
                # name = './fake_lane_' + str(count)
                
                # except:
                #     print('frame error !!!!')
            c += 1
            rval, frame = cap.read()
        cap.release()      

    # f_lane = open('./fake_lane.txt', 'w')
    # for i in range(4):
    #     f_lane.write(str(i) + ',')
    #     f_lane.write(str(a)+' '+str(b)+' ')
    #     for j in range(len(y_sample)-1):
    #         if 0 < x[j] < 1280 and 420 < y_sample[j] < 800 :
    #             f_lane.write(str(x[j])+' '+str(y_sample[j])+' ')
    #     f_lane.write('\n')
    #     f_lane.close()
    cv2.imwrite('./fake_lane.png', img_show_lane)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='The path of config file', default = 'configs/clrnet/clr_resnet34_culane.py')
    parser.add_argument('-i', '--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image', default = False)
    parser.add_argument('-s', '--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='./best_new.pth', help='The path of model')
    args = parser.parse_args()
    process_hough(args)