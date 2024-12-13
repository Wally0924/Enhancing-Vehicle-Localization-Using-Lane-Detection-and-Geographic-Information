import glob ,os, shutil
import cv2
import os
import random
count_train = 0
count_val = 0

if os.path.isdir('./train'):
    shutil.rmtree('./train')
    os.mkdir('./train')
else :
    os.mkdir('./train')

if os.path.isdir('./val'):
    shutil.rmtree('./val')
    os.mkdir('./val')
else :
    os.mkdir('./val')
    


for i in glob.glob('./road_seg_dataset/*.json'):
    # print(i)
    # shutil.move(i, './test/')
    jpg_ = './road_seg_dataset/'+i.split('/')[-1][:-4]+'jpg'
    png_ = './road_seg_dataset/'+i.split('/')[-1][:-4]+'png'
    # # print(jpg_[-3:], png_[-3:])
    if random.randint(0,9)==0:
        shutil.copy(i, './val/')
    # # # print(jpg_, png_)   
        if os.path.isfile(png_): 
            shutil.copy(png_, './val/')

        if os.path.isfile(jpg_):
            shutil.copy(jpg_, './val/')
        count_val += 1
        print('Move ', i , 'to val', count_val)
    else :
        shutil.copy(i, './train/')
        if os.path.isfile(png_): 
            shutil.copy(png_, './train/')

        if os.path.isfile(jpg_):
            shutil.copy(jpg_, './train/')

        count_train += 1    
        print('Move ', i , 'to train', count_train)



