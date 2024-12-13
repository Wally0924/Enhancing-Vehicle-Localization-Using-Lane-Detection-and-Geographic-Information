import osmnx as ox
import numpy as np
from leuvenmapmatching.map.inmem import InMemMap
import pandas as pd
from leuvenmapmatching.matcher.distance import DistanceMatcher
import sys
import logging
import leuvenmapmatching
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import math

def parseDms(lat, latdir, lon, londir):
    deg = int(lat/100)
    seconds = lat - (deg * 100)
    latdec  = deg + (seconds/60)
    if latdir == 'S': latdec = latdec * -1
    deg = int(lon/100)
    seconds = lon - (deg * 100)
    londec  = deg + (seconds/60)
    if londir == 'W': londec = londec * -1
    return round(latdec, 8), round(londec, 8)

def parsetxt(lat, lon):
    lat, latdir = float(lat[1:]), lat[0]
    lon, londir = float(lon[1:]), lon[0]
    return round(lat, 8), round(lon, 8)

def readnmea(nmea):
    gps_all = []
    angle_all = []
    if nmea.split('.')[-1] == 'NMEA':
        with open(nmea , 'r') as f:
            text = f.read().splitlines()
            i = 0
            odo_ini_key = 0
            odo_ini = 0
            node_num = 0
            time_keep = ''

            while i != len(text):
                if text[i].startswith('$GPRMC'):
                    if text[i].split(',')[2] == 'A':
                        odo_ini_key += 1
                        tmp = text[i].split(',')
                        time = tmp[1]
                        if time_keep != time:
                            lat_o, lat_, lon_o, lon_, angle=  tmp[3], tmp[4], tmp[5], tmp[6], tmp[8]
                            lat, lon = parseDms(float(lat_o), lat_, float(lon_o), lon_)
                            print(node_num, lat, lon, 'ori:', lat_o, lon_o)
                            gps_all.append((lat, lon))
                            angle_all.append(float(angle))
                            time_keep = time
                            node_num += 1
                    i += 1
                else:
                    i += 1
        f.close()
    elif nmea.split('.')[-1] == 'txt':
        with open( nmea , 'r') as f:
            text = f.readlines()
            for tex in text:
                tex = tex.strip().split(',')
                lat, lon = tex[1], tex[2]
                lat, lon = parsetxt(lat, lon)
                gps_all.append((lat, lon))
                # angle_all.append(float(angle))

        f.close()
        angle_all = None   
    return gps_all, angle_all

def conver2gpx(gps):
    row_indices = ["lat", "lon"]
    gps = np.array(gps)
    data_df = pd.DataFrame(gps, columns = row_indices)
    Converter.dataframe_to_gpx(input_df=data_df,
                            lats_colname='lat',
                            longs_colname='lon',
                            output_file='test_1.gpx')

    # return data_df
def create_graph(start_node):
    map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)
    graph = ox.graph_from_point(start_node, network_type='drive', dist=1000, simplify=False)
    
    graph_proj = ox.project_graph(graph)
    nodes, edges = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)

    nodes = nodes.to_crs("EPSG:3395")
    edges = edges.to_crs("EPSG:3395")
    # print('======', nodes)
    for nid, row in nodes.iterrows():
        map_con.add_node(nid, (row['lat'], row['lon']))

    for nid1, nid2, _, info in graph.edges(keys=True, data=True):
        map_con.add_edge(nid1, nid2)

    map_con.purge()
    matcher = DistanceMatcher(map_con, max_dist=200, # meter 
                            non_emitting_length_factor=0.75, obs_noise=10, obs_noise_ne=75, # meter dist_noise=10, # meter 
                    non_emitting_edgeid=False)
    return matcher, graph, edges, map_con

def gps_shift(project, node_1_coord, node_2_coord):
    lat = 110936.2
    lon = 101775.45

    vector = [(node_2_coord[0] - node_1_coord[0]) * lat, (node_2_coord[1] - node_1_coord[1]) * lon]

    norm = [vector[1]*-1,vector[0]]
    
    unit_length = math.sqrt(norm[0]*norm[0]+norm[1]*norm[1])
    unitNorm = [norm[0]/unit_length,norm[1]/unit_length]
    dot = unitNorm[0]*vector[0]+unitNorm[1]*vector[1]

    # width = [-5.25, -1.75, 1.75, 5.25]
    width = [-3.5, 3.5]
    lane_xy = []
    for wid in width:
        new_x = project[0] + unitNorm[0]*(wid)/lat
        new_y = project[1] + unitNorm[1]*(wid)/lon
        lane_xy.append([new_x, new_y])
    return lane_xy    


def project_gps_list(matcher, gps_list, map_con, edges, expand = False, th = None): # Enter five gps data

    if th != None:
        th_m = 450 - th
        if th_m > 360:
            th_m = th_m - 360
        vector_gps = np.tan(np.radians(th))
        vector_gps = np.array([1, vector_gps])
    else:
        vector_gps = None

    if expand:
        matcher.match(gps_list, expand=True, vector_gps = vector_gps)
    else:
        matcher.match(gps_list, vector_gps = vector_gps)

    node_1, node_2 = matcher.lattice_best[-1].edge_m.l1, matcher.lattice_best[-1].edge_m.l2
    print(node_1, node_2)

    try:
        match_edge_id =  edges.loc[(node_1, node_2, 0), 'osmid']
    except:
        match_edge_id = 0
    node_1_coord = map_con.node_coordinates(node_1)
    node_2_coord = map_con.node_coordinates(node_2)
    lane_xy = gps_shift(matcher.lattice_best[-1].edge_m.pi, node_1_coord, node_2_coord)

    return matcher, matcher.lattice_best[-1].edge_m.pi, matcher.lattice_best[-1].edge_o.pi, match_edge_id,lane_xy

if __name__ == '__main__' : 
    import time
    import smopy
    import cv2
    import numpy as np
    nmea_path = './GPS_label/201116145511/201116145511.NMEA'
    gps_all, angle_all = readnmea(nmea_path) 
    ori_gps = []
    proj_gps = []

    lane_data = []
    matcher, graph, edges, map_con = create_graph(gps_all[0])
    match_edge_id_keep = 0
    coords = []

    idx = 4
    if angle_all != None:
        matcher, project_point, original_point, match_edge_id,lane_xy = project_gps_list(matcher, gps_all[0:5], map_con, edges, th = angle_all[idx])
    else:
        matcher, project_point, original_point, match_edge_id,lane_xy = project_gps_list(matcher, gps_all[0:5], map_con, edges)
    print(project_point, original_point)

    ori_gps.append(original_point)
    proj_gps.append(project_point)
    lane_data.append(lane_xy)
    idx += 1
    for i in range(idx, len(gps_all)):
        if angle_all != None:
            matcher, project_point, original_point, match_edge_id, lane_xy = project_gps_list(matcher, gps_all[0:i+1], map_con, edges, expand = True, th = angle_all[i])
        else:
            matcher, project_point, original_point, match_edge_id,lane_xy = project_gps_list(matcher, gps_all[0:i+1], map_con, edges, expand = True)
        print(project_point, original_point)
        lane_data.append(lane_xy)
        ori_gps.append(original_point)
        proj_gps.append(project_point)

    def on_press(event):
        if event.button == 1:
            print("my position:" ,event.button,event.ydata, event.xdata)
            ax.scatter(event.xdata,event.ydata, marker='o', s=3, c='white')
            fig.canvas.draw()
        elif event.button == 3:
            x, y  = event.ydata, event.xdata
            global data
            data.append([x, y])
            print('add ground truth node at ',data)
    lane_data = np.array(lane_data)
    
    # pred = []
    # pred_path = open('./GPS_label/201116145511/201116145511_pred_new.txt')
    # pred_data = pred_path.readlines()
    # for i in range(len(pred_data)):
    #     data1, data2 = pred_data[i].strip()[1:-1].split(',')
    #     pred.append([float(data1), float(data2)])
    # pred = np.array(pred)

    for i in range(len(ori_gps)):
        try:
            print(ori_gps[i][1], ori_gps[i][0])
            data = []
            data.append([ori_gps[i][0], ori_gps[i][1]])
            fig, ax = ox.plot_graph(graph, node_color='#999999', show=False, close=False)
            
            ax.plot(lane_data[:,:,1], lane_data[:,:,0], color='white')
            ax.scatter(ori_gps[i][1], ori_gps[i][0], marker='o', s=3, c='red')
            ax.scatter(proj_gps[i][1], proj_gps[i][0], marker='o', s=3, c='green')
            # ax.scatter(pred[i][1], pred[i][0], marker='o', s=3, c='yellow')
            fig.canvas.mpl_connect('button_press_event', on_press)
            plt.show()
            coords.append(data)
        except:
            print('error')
            break

    if nmea_path.split('.')[-1] == 'NMEA':
        txt_path = nmea_path.replace('.NMEA', '_gt.txt')
        f = open(txt_path, 'w')
        for i in range(len(coords)):
            f.write(str(coords[i])+'\n')
        f.close()

    elif nmea_path.split('.')[-1] == 'txt':
        txt_path = nmea_path.replace('.txt', '_gt.txt')
        f = open(txt_path, 'w')
        for i in range(len(coords)):
            f.write(str(coords[i])+'\n')
        f.close()
