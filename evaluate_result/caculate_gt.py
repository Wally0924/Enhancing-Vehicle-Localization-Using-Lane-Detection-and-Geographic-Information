import numpy as np
import math
import cv2
import osmnx as ox
import matplotlib.pyplot as plt


def SE(forecast, prediction):
    xerror = ((forecast[0]-prediction[0])*110936.2)**2
    yerror = ((forecast[1]-prediction[1])*101775.45)**2
    SE = (xerror+yerror)/2
    return SE

def cat_dist(gt, pred):
    #####MAE####
    d_y = abs(gt[:,0]- pred[:,0]) * 110936.2 
    d_x = abs(gt[:,1]- pred[:,1]) * 101775.45  

    dist_x = np.mean(d_x)
    dist_y = np.mean(d_y)
    MAE = round((dist_x + dist_y), 4)

    #####RMSE####
    MSE = 0
    for i in range(len(gt)):
        # MAE = 
        se = SE(gt[i], pred[i])
        MSE += se 
        # print(RMSE_np)
    MSE = (MSE/len(gt) )# * length
    RMSE = round(math.sqrt(MSE), 4)

    return MAE, RMSE

def show_result(gt, gps, pred):
    graph = ox.graph_from_point(gt[0], network_type='drive', dist=1000, simplify=False)
    fig, ax = ox.plot_graph(graph, node_color='#999999', show=False, close=False)

    for i in range(len(gt)):
        ax.scatter(gt[i][1], gt[i][0], marker='o', s=10, c='white')

    for i in range(len(gps)):
        ax.scatter(gps[i][1], gps[i][0], marker='o', s=10, c='red')

    for i in range(len(pred)):
        ax.scatter(pred[i][1], pred[i][0], marker='o', s=10, c='green')
    plt.show()

#gt_path = open('./GPS_label/201116145511/201116145511_gt.txt')
gt_path = open('./ground_truth/localization/201116145511/201116145511_gt.txt')
# ori_pred_path = open('./GPS_label/220531153103/220531153103_original.txt')
pred_path = open('./ground_truth/localization/201116145511/201116145511_pred.txt')

gt_data = gt_path.readlines()
pred_data = pred_path.readlines()
# ori_pred_path = ori_pred_path.readlines()

gt = []
gps = []
pred = []

for i in range(len(gt_data)):
    data1, data2, data3, data4 = gt_data[i].strip()[1:-1].split(',')
    
    data1 = float(data1[1:])
    data2 = float(data2[1:-1])
    data3 = float(data3[2:])
    data4 = float(data4[1:-1])
    gps.append([data1, data2])
    gt.append([data3, data4])

for i in range(len(pred_data)):
    data1, data2 = pred_data[i].strip()[1:-1].split(',')
    pred.append([float(data1), float(data2)])

gps = np.array(gps)
gt = np.array(gt)
pred = np.array(pred)

show_result(gt, gps, pred)
gps_mae, gps_rmse= cat_dist(gt, gps)
pred_mae, pred_rmse = cat_dist(gt, pred)
print(gps_mae, gps_rmse, pred_mae, pred_rmse)

