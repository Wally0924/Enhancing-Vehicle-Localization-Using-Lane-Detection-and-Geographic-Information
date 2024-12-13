import scipy.cluster.hierarchy as sch
import numpy as np
import math

# Post-processing of traffic light data.
# For details, please refer to Section 3.5 of the paper.

def traffic_light_modify(traffic_light_data):
    if traffic_light_data != []:
        traffic_light_data = np.array(traffic_light_data)
        traffic_light_point = traffic_light_data[:,0]
        traffic_node = traffic_light_data[:,1]
        dis=sch.linkage(traffic_light_point, metric='euclidean',method='ward')

        max_dis=50/(math.sqrt(110936.2 *110936.2  +101775.45  *101775.45  ))
        # print(max_dis)
        clusters=sch.fcluster(dis,max_dis,criterion='distance')
        clusters_idx = np.unique(clusters)
        out_traffic = []
        for i in clusters_idx:
            get_node = traffic_node[np.where(clusters == i)]
            unique, counts = np.unique(get_node,axis=0, return_counts=True)
            out_traffic.append(unique[np.where(counts ==np.max(counts))][0])
    else:
        out_traffic = None
    return out_traffic