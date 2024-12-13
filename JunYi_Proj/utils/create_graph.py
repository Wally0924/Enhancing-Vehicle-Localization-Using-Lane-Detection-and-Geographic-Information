from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
import leuvenmapmatching
import osmnx as ox

def create_graph(start_node):
    # Record all nodes and routes within the selected range(here set to 10km)
    # And return matcher(for map matching), graph(for result plot), edges(get information from OSM), map_con(map data structure)

    map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)
    graph = ox.graph_from_point(start_node, network_type='drive', dist=10000, simplify=False)
    graph_proj = ox.project_graph(graph)
    nodes, edges = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)
    
    nodes = nodes.to_crs("EPSG:3395")
    edges = edges.to_crs("EPSG:3395")
    for nid, row in nodes.iterrows():
        map_con.add_node(nid, (row['lat'], row['lon']))
    for nid1, nid2, _, info in graph.edges(keys=True, data=True):
        map_con.add_edge(nid1, nid2)
    map_con.purge()
    matcher = DistanceMatcher(map_con, max_dist=200, # max_dist for searching range. if search failed, try a bigger number.
                            non_emitting_length_factor=0.75, obs_noise=10, obs_noise_ne=75,
                    non_emitting_edgeid=False)
    return matcher, graph, edges, map_con