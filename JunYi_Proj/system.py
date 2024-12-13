import torch
import numpy as np
import cv2
import os
import time 
import osmnx as ox

from yolact_edge.eval import load_yolact
from yolact_edge.eval import savevideo
from yolact_edge.yolact_edge.utils.augmentations import FastBaseTransform

from CLRNet.clrnet.utils.config import Config
from CLRNet.tools.detect import Detect
from CLRNet.clrnet.datasets.process import Process

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import  check_img_size, non_max_suppression
# from yolov5.utils.plots import Annotator, colors
from yolov5.utils.augmentations import letterbox

from utils.fake_lane import create_fake_lane
from utils.load_nmea_data import readnmea
from utils.create_graph import create_graph
from utils.my_process import process_information, process_gps

from skimage.metrics import structural_similarity
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
class system():

    def __init__(self):
        self.device = torch.device("cuda:0")
        self.get_traffic_light = True
        

        print('loading lane detection model...')
        self.lane_cfg = Config.fromfile('./CLRNet/configs/clrnet/clr_resnet34_ceo.py')  # clr_resnet34_160720, clr_resnet34_ceo
        #self.lane_cfg = Config.fromfile('./CLRNet/configs/clrnet/clr_resnet34_160720.py')
        self.lane_detect = Detect(self.lane_cfg)
        self.lane_detect_process = Process(self.lane_cfg.val_process, self.lane_cfg)
        self.lane_detect_cut_height = 590  # 220531, 201116 :590, 160720 : 400
        print('lane detect model load')
        print('loading road detection model...')
        self.road_datect = load_yolact()
        self.road_datect.detect.use_fast_nms = True
        print('road detect model load')

        if self.get_traffic_light:
            
            print('loading traffic light detection model...')
            self.traffic_light_weight = './yolov5/traffic_light_best.pt'
            self.traffic_light_config = './yolov5/data/coco128.yaml'
            self.traffic_light_detect = DetectMultiBackend(self.traffic_light_weight, dnn=True, data=self.traffic_light_config, fp16=True)
            self.traffic_light_detect = self.traffic_light_detect.eval()
            self.traffic_light_stride = self.traffic_light_detect.stride
            self.traffic_light_names = self.traffic_light_detect.names
            self.traffic_light_pt = self.traffic_light_detect.pt
            self.traffic_light_imgsz = check_img_size((640, 640), s=self.traffic_light_stride) 
            print('traffic light detect model load')

        self.y_sample = np.arange(980, 680, -40)# 220531 : (980, 680, -40)  201116 : (880, 580, -40) 160720 : (780, 420, -20)
        self.fake_lane, self.fake_lane_sample, self.exist_lane_data = create_fake_lane('./fake_lane/220531_fake_lane.txt', self.y_sample)
        self.gps_data, self.angle_data = readnmea('./test_data/160720/140707_cut.txt')   
        # print(len(self.gps_data), self.gps_data[-1])
        self.video_path = './test_data/220531/220531153403.MOV'                        # 140707_cut   141713_cut
        self.save_path =  './final_result/'                                                      # 201116145511 201116145712
        self.save_json_path = './final_result/label/'                                            # 220531153103 220531153403
        self.video_start_from = 5
        self.wait_gps_second = 1
        self.wait_gps_flag = True

        self.matcher, self.graph, self.edges, self.map = create_graph(self.gps_data[0])

        self.frame_num = 1
        self.gps_count = -1 
        self.gps_proj_flag = True
        self.gps_expand_flag = False
        self.if_save_all_data = True
        self.crop_img_keep = np.zeros((25, 35))
        self.match_way_id_keep = -1
        self.show = True
        self.save_pic = True
        
        
        self.timeF = 5
        self.video_frame_num = 1
        
        self.new_project = []
        self.original_point_data = []
        self.original_project = []
        self.traffic_light = []

        self.process_information = process_information(self.fake_lane, self.fake_lane_sample, self.exist_lane_data, self.if_save_all_data, self.y_sample)
        self.process_gps = process_gps()

    def warmup(self):
        # In some cases, inferring an empty image with the model first can bring the GPU to a working state.

        x = torch.zeros((1, 3, 320, 800)).to(self.device) + 1
        for _ in range(10):
            self.lane_detect.net(x)
        if self.get_traffic_light:
            self.traffic_light_detect.warmup(imgsz=(1 if self.traffic_light_detect.pt else 1, 3, *self.traffic_light_imgsz)) 
        print('warm up done')

    def gps_diff(self):
        # This is used to judge the change of time. 
        # For video 160720 use line 115, others 114.
        # The reason is two dash cam is different.

        crop_img = cv2.cvtColor(self.frame[990:1015, 615:650], cv2.COLOR_BGR2GRAY)
        # crop_img = cv2.cvtColor(self.frame[15:40, 326:361], cv2.COLOR_BGR2GRAY) # 160720
        self.score, diff = structural_similarity(self.crop_img_keep, crop_img, full = True, data_range=255)
        # print('score', self.score)
        if self.score < 0.9:
            self.gps_count += 1
            # print(self.gps_count)
            self.crop_img_keep = crop_img
            if self.gps_count > self.wait_gps_second and self.wait_gps_flag:
                self.gps_count = 0  # for 201116145511 -> 4
                self.wait_gps_flag = False
                # print('start get gps !!!!!!!!!!!', self.gps_count)
    
    def process_lane(self):
        # This is used to infer the lane detection models.

        img = self.frame[self.lane_detect_cut_height:, :, :]
        lane_data = {'img': img, 'lanes': []}
        lane_data = self.lane_detect_process(lane_data)
        lane_data['img'] = lane_data['img'].unsqueeze(0)
        with torch.no_grad():
            lane_ = self.lane_detect.net(lane_data)
        lane_data['lanes'] = self.lane_detect.net.module.heads.get_lanes(lane_)[0]
        self.lanes = [lane.to_array(self.lane_cfg) for lane in lane_data['lanes']]
        
    def process_road(self):
        # This is used to infer the drivable area detection models.

        road_img = torch.from_numpy(self.frame).cuda().float()
        road_data = FastBaseTransform()(road_img.unsqueeze(0))

        extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}
        self.road = self.road_datect(road_data, extras=extras)["pred_outs"]

    def process_traffic(self):
        # This is used to infer the traffic light detection models.

        img_v5 = letterbox(self.frame, self.traffic_light_imgsz, stride=32, auto=True)[0]
        # Convert
        img_v5 = img_v5.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_v5 = np.ascontiguousarray(img_v5)
        img_v5 = torch.from_numpy(img_v5).to(self.device)
        img_v5 = img_v5.half()
        img_v5 /= 255
        if len(img_v5.shape) == 3:
                img_v5 = img_v5[None] 

        with torch.no_grad():
            pred = self.traffic_light_detect(img_v5, augment=False, visualize=False)
        self.traffic = non_max_suppression(pred, 0.7, 0.45, None, True, max_det=1000)


    def project_gps_list(self, expand = False):
        # This is used to perform map matching.
        # From line 176 to 184. Calculate the angle of the vehicle.
        # From line 186 to 189. Whether to expand the data is determined according to whether the GPS position is the first piece of data.
        # From line 196 to 199. Get the osm id based on the map matching result.
        #                       The default is 0, which means map matching fail.
        # From line 200 to 203. Get the number of lanes based on the map matching result. 
        #                       The default is 0, which means there is no information on the number of lanes.


        if self.angle_data != None:
            th = self.angle_data[self.gps_count-1]
            th_m = 450 - th
            if th_m > 360:
                th_m = th_m - 360
            vector_gps = np.tan(np.radians(th))
            vector_gps = np.array([1, vector_gps])
        else:
            vector_gps = None

        if expand:
            self.matcher.match(self.gps_data[0:self.gps_count], expand=True, vector_gps = vector_gps)
        else:
            self.matcher.match(self.gps_data[0:self.gps_count], vector_gps = vector_gps)

        node_1, node_2 = self.matcher.lattice_best[-1].edge_m.l1, self.matcher.lattice_best[-1].edge_m.l2

        self.proj_node_1 = self.map.node_coordinates(node_1)
        self.proj_node_2 = self.map.node_coordinates(node_2)

        try:
            self.match_way_id = self.edges.loc[(node_1, node_2, 0), 'osmid']
        except:
            self.match_way_id = 0
        try:
            self.lane_num = edges.loc[(node_1, node_2, 0), 'lanes']
        except:
            self.lane_num = 0
        self.original_point =  self.matcher.lattice_best[-1].edge_o.pi
        self.project_point =  self.matcher.lattice_best[-1].edge_m.pi

    def prep_display(self):
        # This is used to process lane, drivable area, gps data. For others information take a look at my_process.py.

        self.process_information.generate_mask(self.road, self.frame)
        self.process_information.lane_exist(self.change_way)
        self.process_information.change_lane()
        self.process_information.cat_lane_available()
        self.process_information.lane_state()
        if len(self.lanes) >=1:
            self.process_information.cat_romve_lane(self.lanes)
            self.process_information.match_pred_virtual()
            self.process_information.modify_exist_virtual()
            self.process_information.remove_lane_stage_1()
            self.process_information.virtual_lane_stage_1()
            self.process_information.virtual_lane_stage_2()
            # if self.process_information.change_lane_flag:
            self.process_information.virtual_lane_stage_3_change_lane()
            # else:
            #     self.process_information.virtual_lane_stage_3()
            self.process_information.remove_lane_stage_2()
        self.process_information.offset_compute()
        self.process_information.draw_result()

    def modify_gps(self):
        # This is used to modify GPS localization by the result from prep_display function.
        # And will only be calculated when the GPS changes.

        if self.score < 0.9:
            self.process_gps.project_modify(self.project_point, self.process_information.offset, self.process_information.state, self.proj_node_1, self.proj_node_2, self.lane_num)
            self.new_project.append(self.process_gps.project)
            self.original_project.append(self.project_point)
            self.original_point_data.append(self.original_point)
            # print('original:', self.original_point)
    def inform_plot(self):
        # This is used to display the results and store.

        ## FPS
        fps = round(1/ (self.end_time - self.start_time), 2)
        print('fps', fps, (self.end_time - self.start_time))
        self.output_img = cv2.putText(self.output_img, 'FPS : ' + str(fps), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        ## trafic light
        if self.get_traffic_light:
            try:
                if len(self.traffic[0]) != 0 :
                    
                    self.output_img = cv2.putText(self.output_img, 'Traffic light : True', (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    if self.if_save_all_data:
                        try:
                            self.traffic_light.append([self.project_point, self.proj_node_2])
                        except:
                            print('traffic ligth no GPS data at', self.frame_num)
                else:
                    self.output_img = cv2.putText(self.output_img, 'Traffic light : False', (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            except:
                self.output_img = cv2.putText(self.output_img, 'Traffic light : False', (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        ## lane
        if self.if_save_all_data:
            from utils.generate_lane_data import LabelFile
            
            label=LabelFile(self.process_information.get_all_lane, os.path.join(self.save_json_path , self.out_file.split('.')[0]+'_'+str(self.frame_num)+'.png'))
            label.load()
            cv2.imwrite(os.path.join(self.save_img_path , self.out_file.split('.')[0]+'_'+str(self.frame_num)+'.png'), self.frame)
            
        if self.show:
            show_img = cv2.resize(self.output_img, (640, 360))
            cv2.imshow('t', show_img)
            cv2.waitKey(1)

        if self.save_pic:
            
            saveimg_dir = os.path.join(self.save_path, self.out_file.split('.')[0])
            if not os.path.exists(saveimg_dir): os.makedirs(saveimg_dir)
            
            cv2.imwrite(os.path.join(saveimg_dir , self.out_file.split('.')[0]+'_'+str(self.frame_num)+'.png'), self.process_information.output_img_noinfo)
        
        self.video_output.write(self.output_img)
        
    def plot_gps_result(self):
        # This is used to display the positioning results and the location of traffic lights.

        fig, ax = ox.plot_graph(self.graph, node_color='#999999', show=False, close=False)

        for i in range(len(self.new_project)):
            # print(type(self.new_project[i][0][1]), self.new_project[i][0][1], self.new_project[i][0][0])
            ax.scatter(self.new_project[i][1], self.new_project[i][0], marker='o', s=3, c='green')

        for i in range(len(self.original_project)):
            # print(type(self.new_project[i][0][1]), self.new_project[i][0][1], self.new_project[i][0][0])
            ax.scatter(self.original_project[i][1], self.original_project[i][0], marker='o', s=3, c='white')

        for i in range(len(self.gps_data)):
            # print('ggg', type(self.gps_data[i][1]),self.gps_data[i][1], self.gps_data[i][0])
            ax.scatter(self.gps_data[i][1], self.gps_data[i][0], marker='o', s=3, c='red')

        if self.get_traffic_light:
            if self.traffic_light != None:
                for i in range(len(self.traffic_light)):
                    ax.scatter(self.traffic_light[i][1], self.traffic_light[i][0], marker='^', s=50, c='yellow', linewidths = 2, edgecolor ="red")
        
        plt.show()
    def save_result_txt(self):
        # This is used to store the GPS location information of each project.

        if self.if_save_all_data:
            
            saveimg_dir = os.path.join(self.save_path, self.out_file.split('.')[0])
            if not os.path.exists(saveimg_dir): os.makedirs(saveimg_dir)

            txt_path = os.path.join(saveimg_dir, self.out_file.split('.')[0]+'_pred.txt')
            f = open(txt_path, 'w')
            for i in range(len(self.new_project)):
                f.write(str(self.new_project[i])+'\n')
            f.close()

            txt_path = os.path.join(saveimg_dir, self.out_file.split('.')[0]+'_original.txt')
            f = open(txt_path, 'w')
            for i in range(len(self.original_point_data)):
                f.write(str(self.original_point_data[i])+'\n')
            f.close()

            txt_path = os.path.join(saveimg_dir, self.out_file.split('.')[0]+'_original_proj.txt')
            f = open(txt_path, 'w')
            for i in range(len(self.original_project)):
                f.write(str(self.original_project[i])+'\n')
            f.close()

            if self.get_traffic_light:
                from utils.traffic_light_modify import traffic_light_modify
                self.traffic_light = traffic_light_modify(self.traffic_light)
                if self.traffic_light != None:
                    txt_path = os.path.join(saveimg_dir, self.out_file.split('.')[0]+'_traffic.txt')
                    f = open(txt_path, 'w')
                    if self.traffic_light:
                        for i in range(len(self.traffic_light)):
                            f.write(str(self.traffic_light[i])+'\n')
                    f.close()

    def process(self):
        # Here is the main program.

        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_file = self.video_path.split('/')[-1].replace('MOV', 'mp4')
        if self.if_save_all_data:
            self.save_json_path = os.path.join(self.save_json_path, self.out_file.split('.')[0],'json')
            if not os.path.exists(self.save_json_path): os.makedirs(self.save_json_path)
            self.save_img_path = self.save_json_path.replace('json', 'img')
            if not os.path.exists(self.save_img_path): os.makedirs(self.save_img_path)

        self.ret, self.frame = cap.read()
        self.video_output = cv2.VideoWriter(os.path.join(self.save_path, self.out_file), fourcc, 5.0, (self.frame.shape[1], self.frame.shape[0])) # 1280, 800 1920, 1080
        while self.ret:
            if self.video_frame_num % self.timeF == 0:
                print('frame_num:', self.frame_num)
                self.start_time = time.time()
                self.gps_diff()
                # start = time.time()
                self.process_road()
                # print('road', time.time()-start)
                # start = time.time()
                self.process_lane()
                # print('lane', time.time()-start)
                
                if self.gps_count > len(self.gps_data):
                    print('done!! ', 'gps count', self.gps_count)
                    break
               
                if self.gps_count >= self.video_start_from : 
                    if self.gps_expand_flag == False:
                        self.project_gps_list()
                        self.gps_expand_flag = True
                    else:
                        # start = time.time()
                        self.project_gps_list(expand = True)
                    if self.get_traffic_light and self.if_save_all_data:
                        # start = time.time()
                        self.process_traffic()
                        # print('process_traffic', time.time()-start)
                    if self.match_way_id_keep != self.match_way_id and self.match_way_id_keep != -1:
                        self.change_way = True
                        self.prep_display()
                        self.match_way_id_keep = self.match_way_id
                        self.modify_gps()
                    else:
                        self.change_way = False
                        # start = time.time()
                        self.prep_display()
                        # print('prep_display', time.time()-start)
                        if self.match_way_id != 0:
                            self.match_way_id_keep = self.match_way_id
                        # start = time.time()
                        self.modify_gps()
                        # print('modify gps', time.time()-start)
                else:
                    self.traffic = [[]]
                    self.change_way = False
                    self.prep_display()

                self.output_img = self.process_information.output_img
                self.end_time = time.time()
                self.inform_plot()
                self.frame_num +=1
            self.video_frame_num += 1
            self.ret, self.frame = cap.read()

        self.save_result_txt()
        self.plot_gps_result()
        
if __name__ == '__main__':
    system = system()
    system.warmup()
    system.process()
    # print(system.fake_lane)
