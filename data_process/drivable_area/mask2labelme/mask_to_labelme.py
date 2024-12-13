import cv2
import numpy as np
import json
import os,io
import glob
import PIL.Image
import base64
import contextlib
import os.path as osp
import shutil, random

class LabelFile(object):

    def __init__(self, filename=None, savefile = None, label_name = None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        # if filename is not None:
        #     self.load(filename)
        self.filename = filename
        self.savefile = savefile
        self.label_name = label_name
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


    def create_json(self):
        filename = self.filename
        savename = self.savefile + '/' + filename.split('/')[-1][:-3]+'json' 
        image_path=filename
        image_path_name=filename.split('/')[-1][:-3]+'png'
        shape=[]
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,gray = cv2.threshold(gray, 36, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        xy = []
        for i in contours:
            a=[]
            for j in i:
                if random.randint(1,8)==1:  # ramdom save 1/8 poins 
                    a.append([float(j[0][0]),float(j[0][1])])
            if len(a)>30:  # Filter too small mask
                xy.append(a)
        
        for i in range(len(xy)):
            shape.append({
                "label": self.label_name,
                "points": xy[i],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
                })

        print(savename)
        self.save(
            filename=savename,
            shapes=shape,
            imagePath=image_path_name,
            imageHeight=image.shape[0],
            imageWidth=image.shape[1],
        )


        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Mask to lebelme json format.'
    )
    parser.add_argument(
        '-m', '--mask', help="mask image in png format", default="./mask", type=str
    )
    parser.add_argument(
        '-l', '--label_name', help="lebel name", default="road", type=str
    )
    parser.add_argument(
        '-o', '--output', help="Output json file path.", default="./ori" , type=str
    )

    args = parser.parse_args()
    for i in glob.glob(args.mask + '/*.png'): # mask place
        label=LabelFile(i, savefile = args.output, label_name = args.label_name)
        label.create_json()

