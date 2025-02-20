import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.spatial import KDTree
from skimage.feature import peak_local_max

def create_label_coordinates(dataPath, shape=(768,768)):
    img_label_other = np.zeros(shape, np.float64)
    img_label_immune = np.zeros(shape, np.float64)
    data = pd.read_csv(dataPath, sep='\t')

    for index, row in data.iterrows():
        #x = int(np.rint(row['x']))-1
        #y = int(np.rint(row['y']))-1

        x = int(np.rint(row['x']/2))-1
        y = int(np.rint(row['y']/2))-1
        
        x = min(x, img_label_other.shape[1])
        x = max(x, 0)
        y = min(y, img_label_other.shape[0])
        y = max(y, 0)

        if row['class'] == 'Stroma':
            img_label_other[y, x] = 1
        elif row['class'] == 'normal':
            img_label_other[y, x] = 1
        elif row['class'] == 'Tumor':
            img_label_other[y, x] = 1
        elif row['class'] == 'Immune cells':
            img_label_immune[y, x] = 1
        elif row['class'] in ['endothelium', 'Endothelial', 'Endothelium']:
            img_label_other[y, x] = 1
        else:
            img_label_other[y, x] = 1
    return img_label_other, img_label_immune


def calculate_estimated_coordinates(pred):
    # get dot predictions (centers of connected components)
    contours, hierarchy = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    
    return (np.array(e_coord_x), np.array(e_coord_y))




def matlab_style_gauss(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def insetGaussian(h_gaussian, e_coordinate, size):
    e_indx_map = np.zeros(size)
    height, width = h_gaussian.shape
    # Define the center coordinates (cx, cy) where h_gaussian should be placed
    cy, cx = e_coordinate
    
    # Compute top-left starting indices
    x_start = cx - width // 2
    y_start = cy - height // 2

    x_end = x_start + width
    y_end = y_start + height

    # Clip indices to stay within bounds
    x_start_clipped = max(0, x_start)
    y_start_clipped = max(0, y_start)
    x_end_clipped = min(size[1], x_end)
    y_end_clipped = min(size[0], y_end)

    # Compute offsets for h_gaussian
    h_x_start = x_start_clipped - x_start  # Offset for source array
    h_y_start = y_start_clipped - y_start
    h_x_end = h_x_start + (x_end_clipped - x_start_clipped)
    h_y_end = h_y_start + (y_end_clipped - y_start_clipped)

    # Ensure matching shapes before assignment
    e_indx_map[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped] = \
        h_gaussian[h_y_start:h_y_end, h_x_start:h_x_end]

    return e_indx_map

def CrowdMatchingTest(g_dot, estimation, sigma_list, sigma_thresh_list, inputType='Segmentation'):
    arr_prec=np.zeros((len(sigma_list), len(sigma_thresh_list)))
    arr_recall=np.zeros((len(sigma_list), len(sigma_thresh_list)))
    arr_f1=np.zeros((len(sigma_list), len(sigma_thresh_list)))

    if inputType=='Segmentation':
        e_coordinate = calculate_estimated_coordinates(estimation)
        e_coord_x, e_coord_y = e_coordinate
    elif inputType=='Regression':
        estimation[estimation<0.001]=0        
        e_coord = peak_local_max(estimation, min_distance=3)
        e_coord_x = e_coord[:, 1]
        e_coord_y = e_coord[:, 0]
    elif inputType=='Coordinates':
        e_coordinate = estimation
        e_coord_x, e_coord_y = e_coordinate
    else:
        print('INVALID Input')
        
    g_count = np.sum(g_dot)
    
    g_count = np.sum(g_dot)
    if g_count == 0:
        if len(e_coord_x) == 0:
            arr_prec.fill(1)
            arr_recall.fill(1)
            arr_f1.fill(1)
            return arr_prec, arr_recall, arr_f1
        else:
            arr_recall.fill(1)
            return arr_prec, arr_recall, arr_f1
    ### iterate over sigma(gauss)
    for s in range(len(sigma_list)):
        sigma = sigma_list[s]
        
        # Calculate kernel size
        radius = int(round(4 * sigma))
        ksize = (2 * radius + 1, 2 * radius + 1)
        h_gaussian = matlab_style_gauss(shape=ksize,sigma=sigma)
        sigma_max = h_gaussian.max()
        # iterate over threshold
        for t in range(len(sigma_thresh_list)):
            thresh = sigma_thresh_list[t]
            
            tp = 0
            fp = 0
            fn = 0
            g_dot_remaining = g_dot.copy()

            # iterate over estimations
            for e_indx in range(len(e_coord_y)):
                # # create a map that is all zeros except one at the current prediction center
                # e_indx_map[e_coord_y[e_indx], e_coord_x[e_indx]] = 1  
                # import a gaussian kernel centered at this point with maximum value 1
                et_sigma = insetGaussian(h_gaussian, (e_coord_y[e_indx], e_coord_x[e_indx]), (g_dot.shape))/sigma_max  

                # element-wise multiply et_sigma with the gt point map
                gt_sigma = et_sigma * g_dot_remaining
                # get max value which corresponds to closest gt
                g_closest_val = gt_sigma.max()
    
                # find if true positive based on current threshold
                if(g_closest_val < thresh):
                    fp += 1
                else:
                    tp += 1
                    # exclude matched point from ground truth map so that it is not matched again.
                    g_y, g_x = np.where(gt_sigma == g_closest_val)
                    g_dot_remaining[g_y[0], g_x[0]] = 0
    
            # false negatives are remaining dots in ground truth map that were not matched.
            fn = g_count - tp
            if(fn < 0):
                fn = 0    
            prec = tp / (tp + fp+1e-7)
            recall = tp/ (tp + fn)
            f1score = 2*prec*recall/(prec+recall+1e-7)
            arr_prec[s,t] =  prec
            arr_recall[s,t] = recall
            arr_f1[s,t] = f1score
            
    return arr_prec, arr_recall, arr_f1

def CrowdMatchingTest2(g_dot, estimation, sigma_list, sigma_thresh_list, inputType='Segmentation'):
    arr_prec=np.zeros((len(sigma_list), len(sigma_thresh_list)))
    arr_recall=np.zeros((len(sigma_list), len(sigma_thresh_list)))
    arr_f1=np.zeros((len(sigma_list), len(sigma_thresh_list)))

    if inputType=='Segmentation':
        e_coordinate = calculate_estimated_coordinates(estimation)
    elif inputType=='Regression':
        print('TODO')
    elif inputType=='Coordinates':
        e_coordinate = estimation
    else:
        print('INVALID Input')
        
    e_coord_x, e_coord_y = e_coordinate
    g_count = np.sum(g_dot)
    
    g_count = np.sum(g_dot)
    if g_count == 0:
        if len(e_coord_x) == 0:
            arr_prec.fill(1)
            arr_recall.fill(1)
            arr_f1.fill(1)
            return arr_prec, arr_recall, arr_f1
        else:
            arr_recall.fill(1)
            return arr_prec, arr_recall, arr_f1
    ### iterate over sigma(gauss)
    for s in range(len(sigma_list)):
        sigma = sigma_list[s]
        
        # Calculate kernel size
        radius = int(round(4 * sigma))
        ksize = (2 * radius + 1, 2 * radius + 1)
        h_gaussian = matlab_style_gauss(shape=ksize,sigma=sigma)
        sigma_max = h_gaussian.max()
        # iterate over threshold
        for t in range(len(sigma_thresh_list)):
            thresh = sigma_thresh_list[t]
            
            tp = 0
            fp = 0
            fn = 0
            g_dot_remaining = g_dot.copy()

            # iterate over estimations
            for e_indx in range(len(e_coord_y)):
                # # create a map that is all zeros except one at the current prediction center
                # e_indx_map[e_coord_y[e_indx], e_coord_x[e_indx]] = 1  
                # import a gaussian kernel centered at this point with maximum value 1
                et_sigma = insetGaussian(h_gaussian, (e_coord_y[e_indx], e_coord_x[e_indx]), (g_dot.shape))/sigma_max  

                # element-wise multiply et_sigma with the gt point map
                gt_sigma = et_sigma * g_dot_remaining
                # get max value which corresponds to closest gt
                g_closest_val = gt_sigma.max()
    
                # find if true positive based on current threshold
                if(g_closest_val < thresh):
                    fp += 1
                else:
                    tp += 1
                    # exclude matched point from ground truth map so that it is not matched again.
                    g_y, g_x = np.where(gt_sigma == g_closest_val)
                    g_dot_remaining[g_y[0], g_x[0]] = 0
    
            # false negatives are remaining dots in ground truth map that were not matched.
            fn = g_count - tp
            if(fn < 0):
                fn = 0    
            prec = tp / (tp + fp)
            recall = tp/ (tp + fn)
            f1score = 2*prec*recall/(prec+recall+1e-7)
            arr_prec[s,t] =  prec
            arr_recall[s,t] = recall
            arr_f1[s,t] = f1score
            
    return arr_prec, arr_recall, arr_f1

def countAccuracyMetric(countGT, countPred):
    '''
    Takes two integer: ground-truth and estimated count
    Returns 4 accuracy metrics
    '''
    abs_diff = abs(countGT-countPred)
    Accuracy = round(abs_diff/(countGT+1e-6),4)
    AccuracyRelative = round(abs_diff/(max(countGT,countPred)+1e-6),4)
    AccuracyRelativePD =round((2*abs_diff)/(countGT+countPred+1e-6),4)
    return abs_diff, Accuracy, AccuracyRelative, AccuracyRelativePD

def GMAE(L, gtImg, predImg):
    # Calculate the number of cells
    num_cells = 4 ** L
    cell_size = 768 // (2 ** L)
    
    # Initialize a list to store the cells
    gmae_cell = 0
    gmae_cellAccuracyRelative = 0
    gmae_cellAccuracyRelativePD = 0

    # Iterate over the image and extract cells
    for i in range(0, 768, cell_size):
        for j in range(0, 768, cell_size):
            current_GT = gtImg[i:i+cell_size, j:j+cell_size]
            current_pred = predImg[i:i+cell_size, j:j+cell_size]
            cellCountGt = np.sum(current_GT)
            cellCountPred = np.sum(current_pred)                
            abs_diff_cell, _, cellAccuracyRelative, cellAccuracyRelativePD = countAccuracyMetric(cellCountGt, cellCountPred)
            gmae_cell += abs_diff_cell
            gmae_cellAccuracyRelative += cellAccuracyRelative
            gmae_cellAccuracyRelativePD += cellAccuracyRelativePD

    return [gmae_cell, gmae_cellAccuracyRelative, gmae_cellAccuracyRelativePD]

def main():

    path = '/home/ocaki13/projects/topology/goodExample/20-8545-AI1-2_1319_827.png_pred.png'
    pred = cv2.imread(path,0)
    path = '/home/ocaki13/projects/topology/goodExample/20-8545-AI1-2_1319_827_label_mc.png'
    gt = cv2.imread(path,0)

    gt_dot_other, gt_dot_immune = create_label_coordinates('/home/ocaki13/projects/topology/goodExample/20-8545-AI1-2_1319_827.png-points.tsv')
    pred_other = np.zeros_like(pred)
    pred_immune = np.zeros_like(pred)
    pred_other[pred==1] = 1
    pred_immune[pred==2] = 1

    sigma_list=[5, 20]
    sigma_thresh_list=list(np.arange(0.5,1, 0.05))

    # total_time = 0
    # start = time.time()
    # arr_prec_current, arr_recall_current, arr_f1_current = CrowdMatchingTest(gt_dot_immune, pred_immune, sigma_list, sigma_thresh_list)
    # stop = time.time()
    # total_time = stop - start
    # print("Time {} sec".format(round(total_time/60, 4)))
    # print(np.mean(arr_prec_current, axis=1))
    # print(np.mean(arr_recall_current, axis=1))
    # print(np.mean(arr_f1_current, axis=1))

    # print(np.mean(arr_prec_current[:,:-1], axis=1))
    # print(np.mean(arr_recall_current[:,:-1], axis=1))
    # print(np.mean(arr_f1_current[:,:-1], axis=1))
    
    total_time = 0
    start = time.time()
    prec, recall, f1 = GreedyMatchingTest(gt_dot_immune, pred_immune)
    stop = time.time()
    total_time = stop - start
    print(prec)
    print(recall)
    print(f1)

    
if __name__ == '__main__':
    main()
