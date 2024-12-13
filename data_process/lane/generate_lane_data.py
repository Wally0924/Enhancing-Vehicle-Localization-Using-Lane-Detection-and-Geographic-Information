import cv2
import numpy as np
import json
import os,io
import glob
import PIL.Image
import base64
import contextlib
import os.path as osp
import shutil


class LabelFile(object):

    def __init__(self, lane=None, filename = None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        # if filename is not None:
        #     self.load(filename)
        self.lane = lane
        self.filename = filename
    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):

        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version="4.5.7",
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.filename = filename


    def load(self):
        # line = open(self.filename)
        # print(self.filename)
        savename=self.filename.replace('png', 'json')
        # data_annotation=json.load(line)
        # image_path_name=self.filename.strip().split('\\')[-1][:-4]+'png'
        # image_path=self.filename.replace('json','png').replace('Annotation2','train')
        
        # shutil.copyfile(image_path,'./labelme_point/'+image_path_name)
        temp_lanes = []
        shape=[]
        # img=cv2.imread(image_path)
        # print(img.shape[0],img.shape[1])
        # print(image_path_name)
        for i in range(4):
            if str(i) in self.lane.keys():
                shape.append({
                    "label": 'lane',
                    "points": self.lane[str(i)],
                    "group_id": None,
                    "shape_type": "linestrip",
                    "flags": {}
                    })
        self.save(
            filename=savename,
            shapes=shape,
            imagePath=self.filename.split('/')[-1],
            imageHeight=1080,
            imageWidth=1920,
        )


        
if __name__ == '__main__':

    lane = {'1':[[ 428.41903891,1060.0],
                [ 455.22345679,1040.0],
                [ 482.018389,1020.0],
                [ 508.86602071,1000.0],
                [ 535.71536626,980.0]],
            '0':[[ 62.61804679,860.0],
                [164.12353326,840.0],
                [264.81506011,820.0],
                [363.15752456,800.0],
                [460.12218766,780.0]]}
    label=LabelFile(lane, '220531153403_1.png')
    label.load()