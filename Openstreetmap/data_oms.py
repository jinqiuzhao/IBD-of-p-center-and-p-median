import osmnx as ox
import numpy as np
import networkx as nx
import os
from IPython.display import Image

if __name__ == '__main__':
    # place = {
    #     "city": "shenzhen",
    #     # "county": "Alameda County",
    #     # "state": "广东",
    #     "country": "China",
    # }
    # place = "福田区, 深圳, 中国"
    # place = "南山区, 深圳, 中国"
    # place = "北京, 中国"
    # place = "成都市, 四川省, 中国"
    # place = "Piedmont, California, USA" # dist 2500, random
    # place = "Dubai, United Arab emirates"
    # place = 'Manhattan, New York, USA' # dist 3000, rectangle
    place = "Portland, USA"  # dist 2000, rectangle,
    # place = "San Francisco, USA"
    # boundaries = ox.geocode_to_gdf(place)
    city = ox.geocode_to_gdf(place)
    north, south, east, west = city["bbox_north"].iloc[0], city["bbox_south"].iloc[0], \
                               city["bbox_east"].iloc[0], city["bbox_west"].iloc[0]
    location_point = ((north + south) / 2, (east + west) / 2)
    cf = '["highway"~"primary|primary_link|motorway|motorway_link|trunk|trunk_link"]'
    # cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link"]'
    # G = ox.graph_from_place(place, network_type="drive", custom_filter=cf)
    # G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    G = ox.graph_from_point(location_point, dist=2000, dist_type='bbox', network_type="drive")
    # G = ox.graph_from_address(address="天安门广场", dist=10000, dist_type='bbox', network_type="drive", custom_filter=cf)
    # G = ox.graph_from_address(address="天府广场", dist=10000, dist_type='bbox', network_type="drive", custom_filter=cf)
    fig, ax = ox.plot_graph(G, node_size=0, edge_color="w", edge_linewidth=0.2, dpi=600, save=True)
    # fp = f"beijing.png"
    # fig, ax = ox.plot_figure_ground(
    #     G, filepath=fp, dpi=100, save=True, show=False, close=True
    # )
    # Image(fp, height=2400, width=2400)

    D = ox.utils_graph.get_digraph(G)
    print(D.number_of_nodes())
    print(D.number_of_edges())
    M = ox.utils_graph.get_undirected(G)
    print(M.number_of_nodes())
    print(M.number_of_edges())

    # path = f'chengdu.npy'
    # if not os.path.isfile(path):
    #     dis_matrix = np.zeros([M.number_of_nodes(), M.number_of_nodes()], dtype=np.int32)
    #     for i in range(M.number_of_nodes()):
    #         for j in range(M.number_of_nodes()):
    #             if i < j:
    #                 orig = list(M)[i]
    #                 dest = list(M)[j]
    #                 route = nx.shortest_path_length(M, orig, dest, weight="length")
    #                 dis_matrix[i, j] = dis_matrix[j, i] = int(route)
    #     np.save("chengdu.npy", dis_matrix)
    # else:
    #     dis_matrix = np.load(path)
    # print(dis_matrix)
    # a = 3

    path = f'Portland.npy'
    if not os.path.isfile(path):
        A = np.array(nx.adjacency_matrix(M).todense())
        dis_matrix = np.ones((A.shape[0], A.shape[1])) * np.inf
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i < j and A[i, j] != 0:
                    orig = list(M)[i]
                    dest = list(M)[j]
                    route = nx.shortest_path_length(M, orig, dest, weight="length")
                    dis_matrix[i, j] = int(route)
                    dis_matrix[j, i] = dis_matrix[i, j]
                elif i == j:
                    dis_matrix[i, j] = 0
        for k in range(A.shape[0]):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    dis_matrix[i, j] = min(dis_matrix[i, j], dis_matrix[i, k] + dis_matrix[k, j])
        np.save(path, dis_matrix)
    else:
        dis_matrix = np.load(path)
    print(dis_matrix)


