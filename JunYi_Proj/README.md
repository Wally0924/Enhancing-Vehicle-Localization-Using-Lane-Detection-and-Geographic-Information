# JuneProj


## Installation

- Python >= 3.8.
- Install Pytorch 1.8.0 and TorchVision.
- Install TensorRT.
- Install other packages.
```python=
pip install cython numpy PyYAML requests scipy tqdm protobuf osmnx addict yapf mmcv imgaug seaborn
pip install opencv-python pillow matplotlib scikit-learn p_tqdm
pip install git+https://github.com/haotian-liu/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
pip install GitPython termcolor tensorboard
pip install leuvenmapmatching
```
- Install Install torch2trt.
```python=
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install --plugins
```
- Install CLRNet dependence
```python=
cd ../CLRNet
python setup.py build develop
```
- Replace the code in the leuvenmapmatching file with the code in map_matching_code file
- Download weights and test data from [here](https://drive.google.com/drive/folders/1TMY9emnEqHV85pAv6ALrabRSt_AUsX6y?usp=sharing).
 - Your folder should be organized like this:
  ```
  JuneProj
  ├── map_matching_code
  ├── fake_lane
  ├── utils
  ├── CLRNet
  │   ├── clrnet_lane.pth
  ├── yolact_edge
  │   ├── weights
  │   │   ├── yolact_edge_resnet50_1467_800000.pth
  │   ├── yolact_edge
  │   │   ├── data
  │   │   │   └── calib
  ├── yolov5
  │   └── traffic_light_best.pt
  └── system.py

  ```
## Run project
```python=
python system.py
```
### suburb
```python=
# video name: 220531153103.MOV, 220531153403.MOV
# fake lane: 220531_fake_lane.txt

# line 39
self.lane_cfg = Config.fromfile('./CLRNet/configs/clrnet/clr_resnet34_ceo.py')
# line 42  
self.lane_detect_cut_height = 590
# line 62
self.y_sample = (980, 680, -40)
# line 63 fake lane 
self.fake_lane, self.fake_lane_sample, self.exist_lane_data = create_fake_lane('./fake_lane/220531_fake_lane.txt', self.y_sample) 
# line 66 GPS info
self.gps_data, self.angle_data = readnmea('./test_data/220531/220531153403.NMEA') 
# line 68 video
self.video_path = './test_data/220531/220531153403.MOV'
```
### urban
```python=
# video name: 201116145511.MOV, 201116145712.MOV
# fake lane: 201116_fake_lane.txt

# line 39
self.lane_cfg = Config.fromfile('./CLRNet/configs/clrnet/clr_resnet34_ceo.py')
# line 42  
self.lane_detect_cut_height = 590
# line 62
self.y_sample = (880, 580, -40)
# line 63 fake lane 
self.fake_lane, self.fake_lane_sample, self.exist_lane_data = create_fake_lane('./fake_lane/220531_fake_lane.txt', self.y_sample) 
# line 66 GPS info
self.gps_data, self.angle_data = readnmea('./test_data/201116/201116145511.NMEA') 
# line 68 video
self.video_path = './test_data/201116/201116145511.MOV'

###!!!!attention!!!!###
when test video is 201116145511.MOV
# line 123
self.gps_count = 4
######################
```
### Highway
```python=
# video name: 140707_cut.mp4, 141713_cut.mp4
# fake lane: 160720_fake_lane.txt

# line 39
self.lane_cfg = Config.fromfile('./CLRNet/configs/clrnet/clr_resnet34_160720.py')
# line 42  
self.lane_detect_cut_height = 400
# line 62
self.y_sample = (780, 420, -20)
# line 63 fake lane 
self.fake_lane, self.fake_lane_sample, self.exist_lane_data = create_fake_lane('./fake_lane/160720_fake_lane.txt', self.y_sample) 
# line 66 GPS info
self.gps_data, self.angle_data = readnmea('./test_data/160720/140707_cut.txt') 
# line 68 video
self.video_path = './test_data/160720/140707_cut.mp4'
# line 203 change lane num
self.lane_num = 3 # default is 0 in project will be considered 2.
```
