import cv2
from skimage.feature import peak_local_max
import time
import numpy as np
from gtda.homology import WeakAlphaPersistence
from gtda.diagrams import PairwiseDistance
import networkx as nx
from scipy.spatial import Delaunay
import math
import ripser
import persim

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extractPoints(BinarySegMap):
    # get dot predictions (centers of connected components)
    contours, hierarchy = cv2.findContours(BinarySegMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    e_coord_y = []
    e_coord_x = []
    for idx in range(len(contours)):
        #print('idx=',idx)
        contour_i = contours[idx]
        M = cv2.moments(contour_i)
        #print(M)
        if(M['m00'] == 0):
            continue;
        cx = round(M['m10'] / M['m00'])
        cy = round(M['m01'] / M['m00'])
        e_coord_y.append(cy)
        e_coord_x.append(cx)

    e_coord = np.zeros((len(e_coord_y),2))
    e_coord[:,0] = np.array(e_coord_x)
    e_coord[:,1] = np.array(e_coord_y)
    return e_coord

def PointCloudFiltration(SegMapGold, SegMapPred, distance='wasserstein',p=2):
    #extract other and immune cell coorditanes for label    
    coordinates_pred = extractPoints(SegMapPred)
    if len(coordinates_pred) < 5:
        return 0
    coordinates_gold = extractPoints(SegMapGold)
    if len(coordinates_gold) < 5:
        return 0
    #filtration
    VR = WeakAlphaPersistence(homology_dimensions=[0, 1], max_edge_length=50)
    inputData = [coordinates_gold, coordinates_pred]
    PDdiagrams = VR.fit_transform(inputData)

    #calculate distance between diagrams
    PD = PairwiseDistance(metric=distance,
                          order=None, metric_params={'p':p})
    distance = PD.fit_transform(PDdiagrams)
    

    return np.mean(distance[1][0])

def GrapEdgeFiltration1(SegMapGold, SegMapPred, distance='wasserstein'):
    #extract other and immune cell coorditanes for label
    coordinates_gold = extractPoints(SegMapGold)
    coordinates_pred = extractPoints(SegMapPred)

    # Construct graph using Delaunay triangulation where edges are distance
    Graph_gold = constructWeightedGrap(coordinates_gold)
    Graph_pred = constructWeightedGrap(coordinates_pred)

    #convert graphs into sparce matrices
    Graph_mat_gold = nx.to_scipy_sparse_array(Graph_gold)
    Graph_mat_pred = nx.to_scipy_sparse_array(Graph_pred)
    

    #compute tersistance diagram using edge filtration
    PD_gold = ripser.ripser(Graph_mat_gold, distance_matrix=True,thresh=50)['dgms']
    PD_pred = ripser.ripser(Graph_mat_pred, distance_matrix=True,thresh=50)['dgms']

    #wassertein distance
    if distance == 'wasserstein':
        distance_H0 = persim.wasserstein(PD_gold[0], PD_pred[0], matching=False)
        distance_H1 = persim.wasserstein(PD_gold[1], PD_pred[1], matching=False)
        
    elif distance == 'image':
        print('TODO')
    else:
        return 'invalid'

    
    return [distance_H0, distance_H1]

def constructWeightedGrap(coordinates):
    # Step 1: Perform Delaunay triangulation
    tri = Delaunay(coordinates)
    
    # Step 2: Create a NetworkX graph
    G = nx.Graph()
    
    # Step 3: Add edges from the Delaunay triangulation
    # The triangulation has simplices (triangles) of point indices
    for simplex in tri.simplices:
        # Each triangle simplex has 3 vertices, so we add 3 edges
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                if not G.has_edge(u, v):
                    # Calculate Euclidean distance between points u and v
                    distance = euclidean_distance(coordinates[u], coordinates[v])
                    # Add edge with weight (distance)
                    G.add_edge(u, v, weight=distance)

    return G

# compute shortest path matrix
def get_path_distance_matrix(G, lnorm='L1'):
    L0 = lnorm == 'L0'
    L1 = lnorm == 'L1'
    node_idx = {n: i for i, n in enumerate(G.nodes)}
    D = np.full((len(G), len(G)), np.inf)
    for n, (dists, paths) in nx.all_pairs_dijkstra(G, weight='weight'):
        for k, dist in dists.items():
            D[node_idx[n], node_idx[k]] = dist if L1 else len(paths[k]) if L0 else 1
    return np.array(np.round(D, 4))

def GrapEdgeFiltrationShortestPath(SegMapGold, SegMapPred, distance='wasserstein'):
    #extract other and immune cell coorditanes for label
    coordinates_gold = extractPoints(SegMapGold)
    coordinates_pred = extractPoints(SegMapPred)

    # Construct graph using Delaunay triangulation where edges are distance
    Graph_gold = constructWeightedGrap(coordinates_gold)
    Graph_pred = constructWeightedGrap(coordinates_pred)


    #convert graphs into sparce matrices
    Graph_mat_gold = get_path_distance_matrix(Graph_gold, lnorm='L1')
    Graph_mat_pred = get_path_distance_matrix(Graph_pred, lnorm='L1')
    
    #compute tersistance diagram using edge filtration
    PD_gold = ripser.ripser(Graph_mat_gold, distance_matrix=True,thresh=50)['dgms']
    PD_pred = ripser.ripser(Graph_mat_pred, distance_matrix=True,thresh=50)['dgms']

    #wassertein distance
    if distance == 'wasserstein':
        distance_H0 = persim.wasserstein(PD_gold[0], PD_pred[0], matching=False)
        distance_H1 = persim.wasserstein(PD_gold[1], PD_pred[1], matching=False)
        
    elif distance == 'image':
        print('TODO')
    else:
        return 'invalid'

    
    return [distance_H0, distance_H1]

def main():

    ge_label_path = 'goodExample/20-8545-AI1-2_1319_827_label_mc.png'
    ge_label = cv2.imread(ge_label_path,0)

    ge_pred_path = 'goodExample/20-8545-AI1-2_1319_827.png_pred.png'
    ge_pred = cv2.imread(ge_pred_path,0)

    be_label_path = 'badExample/20-4321-a1-4_995_671_label_mc.png'
    be_label = cv2.imread(be_label_path,0)
    be_pred_path = 'badExample/20-4321-a1-4_995_671.png_pred.png'
    be_pred = cv2.imread(be_pred_path,0)
    
    total_time = 0
    for i in range(15):
        start = time.time()
        other_map_pred = np.zeros_like(ge_pred)
        other_map_pred[ge_pred == 1] = 1
        immune_map_pred = np.zeros_like(ge_pred)
        immune_map_pred[ge_pred == 2] = 1
        
        other_map_gold = np.zeros_like(ge_label)
        other_map_gold[ge_label == 1] = 1
        immune_map_gold = np.zeros_like(ge_label)
        immune_map_gold[ge_label == 2] = 1
               
        #dist_good_other = PointCloudFiltration(other_map_gold, other_map_pred, 'betti',p=2)
        #dist_good_other = GrapEdgeFiltration1(other_map_gold, other_map_pred)
        dist_good_other = GrapEdgeFiltrationShortestPath(other_map_gold, other_map_pred)

        
        #dist_good_immune = PointCloudFiltration(immune_map_gold, immune_map_pred, 'betti',p=2)
        #dist_good_immune = GrapEdgeFiltration1(immune_map_gold, immune_map_pred)
        dist_good_immune = GrapEdgeFiltrationShortestPath(immune_map_gold, immune_map_pred)

        stop = time.time()
        current_time = stop - start
        total_time +=current_time
    total_time = total_time/15
    print("Time {} ms".format(round(total_time*1000, 4)))
    print(dist_good_other)
    print(dist_good_immune)

    
    total_time = 0
    for i in range(15):
        start = time.time()
        other_map_pred = np.zeros_like(be_pred)
        other_map_pred[be_pred == 1] = 1
        immune_map_pred = np.zeros_like(be_pred)
        immune_map_pred[be_pred == 2] = 1
        
        other_map_gold = np.zeros_like(be_label)
        other_map_gold[be_label == 1] = 1
        immune_map_gold = np.zeros_like(be_label)
        immune_map_gold[be_label == 2] = 1
               
        #dist_good_other = PointCloudFiltration(other_map_gold, other_map_pred, 'betti',p=2)
        #dist_good_other = GrapEdgeFiltration1(other_map_gold, other_map_pred)
        dist_good_other = GrapEdgeFiltrationShortestPath(other_map_gold, other_map_pred)

        #dist_good_immune = PointCloudFiltration(immune_map_gold, immune_map_pred, 'betti',p=2)
        #dist_good_immune = GrapEdgeFiltration1(immune_map_gold, immune_map_pred)
        dist_good_immune = GrapEdgeFiltrationShortestPath(immune_map_gold, immune_map_pred)
        
        stop = time.time()
        current_time = stop - start
        total_time +=current_time
    total_time = total_time/15
    print("Time {} ms".format(round(total_time*1000, 4)))
    print(dist_good_other)
    print(dist_good_immune)

    
if __name__ == '__main__':
    main()
