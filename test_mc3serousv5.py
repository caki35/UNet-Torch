import numpy as np
import cv2
import os
from tqdm import tqdm
import re
import torch
from Model import UNet, UNet_attention
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
from skimage.color import rgb2hed
from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from scipy.spatial import distance
import seaborn as sns
import staintools
from scipy.stats import pearsonr
from CrowdMatching import CrowdMatchingTest, GMAE, countAccuracyMetric
image_ext = ['.png', '.jpg', '.tif', '.tiff']


REFERENCE_PATH = '/home/ocaki13/UNet-Torch/color_normalizer.npy'
REF = np.load(REFERENCE_PATH)

NORMALIZER = staintools.StainNormalizer(method='macenko')
NORMALIZER.fit(REF)

def NoiseFiltering(img, thresh=150):
    unique_labels = np.unique(img)
    for i in unique_labels:
        if i==0:
            continue

        binary_img = np.zeros_like(img)
        binary_img[img==i] = 1
        label_img = label(binary_img)
        label_list = list(np.unique(label_img))
        for lbl in label_list:
            if (label_img == lbl).sum() < thresh:
                img[label_img == lbl] = 0
    return img

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


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            if '_label' not in filename:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_names.append(apath)
    return natural_sort(image_names)


def preprocess(img_org, input_size):        
    if len(img_org.shape)==2:
        imgHeight, imgWidth = img_org.shape
        if imgHeight != input_size[0] or imgWidth != input_size[1]:
            img_input = zoom(img_org, (input_size[0] / imgHeight, input_size[1] / imgWidth), order=3)  
        else:
            img_input = img_org
    else:
        imgHeight, imgWidth, _ = img_org.shape
        if imgHeight != input_size[0] or imgWidth != input_size[1]:
            img_input = zoom(img_org, (input_size[0] / imgHeight, input_size[1] / imgWidth, 1), order=3)  
        else:
            img_input = img_org
        
    
    #z normalizization
    mean3d = np.mean(img_input, axis=(0,1))
    std3d = np.std(img_input, axis=(0,1))
    img_input = (img_input-mean3d)/std3d
    
    if len(img_org.shape)==2:
        img_input = torch.from_numpy(img_input.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    else:  
        # HWC to CHW, BGR to RGB (for three channel)
        img_input = img_input.transpose((2, 0, 1))[::-1]
        img_input = torch.from_numpy(img_input.astype(np.float32)).unsqueeze(0).cuda()
    
    return img_input



class Results3Class:
    def __init__(self, save_dir, iou_thresh = 0.5):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.iou_thresh = iou_thresh
        self.smoothening_factor = 1e-6
        self.recall = []
        self.precision = []
        self.f1 = []
        self.performace_results = {}
        self.classDict =  {1:'other',
                            2:'immune',
                            3:'tumor'}
        self.cellCounts = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.immuneCounts = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.tumorCounts = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.ratio = {'GTImmo':[],'PredImmo':[], 
                      'GTImmoTummor':[], 'PredImmoTummor':[],
                      'AccuracyImmoTummor':[], 'AccuracyImmo':[]}
        self.classRes = {}
        self.imageNames = []
        self.label_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in self.classDict:
            self.classRes[i] = {'tp':0,'fp':0,'fn':0,'tn':0}
        self.epsilon = 1e-9
        self.save_dir = save_dir
        

    def _findObjects(self, img):
        objectDict = {}
        clsList = np.unique(img)
        offset= 0
        cellCounts = {}
        for cls in self.classDict:
            if cls==0:
                continue
            cellCounts[cls] = 0
        for cls in clsList:
            if cls==0:
                continue
            imgCls = np.zeros_like(img)
            imgCls[img==cls] = 1
            contours, _ = cv2.findContours(imgCls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cellCounts[cls] = len(contours)
            for i,contour in enumerate(contours):
                objectDict[i+offset] = {'contour':contour,'class':cls}
            offset = offset+i+1
        return objectDict, cellCounts[1], cellCounts[2], cellCounts[3]


    def _create_rgb_mask(self, mask):
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[mask == 1] = self.label_colors[0]
        rgb_mask[mask == 2] = self.label_colors[1]
        rgb_mask[mask == 3] = self.label_colors[2]

        return rgb_mask
    
    
    def _save_visuals(self, img_org, mask_img, prediction, cellCountsGT, cellCountsPred, save_dir):
        cellCountsGtOthers, cellCountsGtImmune = cellCountsGT
        cellCountsPredOther, cellCountsPredImmune = cellCountsPred
        
        fig, axs = plt.subplots(1, 3)
        fig.set_figheight(12)
        fig.set_figwidth(30)
        if len(img_org.shape) == 3:
            axs[0].imshow(img_org)
            axs[0].title.set_text('image')
        else:
            axs[0].imshow(img_org, cmap='gray')
            axs[0].title.set_text('image')
        axs[1].imshow(self._create_rgb_mask(mask_img))
        axs[1].title.set_text('label')
        fig.text(.51, .17, "tumor: {}".format(cellCountsGtOthers), ha='center',color="red")
        fig.text(.51, .15, "immune {}".format(cellCountsGtImmune), ha='center',color="green")
        axs[2].imshow(self._create_rgb_mask(prediction))
        axs[2].title.set_text('prediction')
        fig.text(.79, .17, "tumor: {}".format(cellCountsPredOther), ha='center',color="red")
        fig.text(.79, .15, "immune {}".format(cellCountsPredImmune), ha='center',color="green")
        fig.savefig(os.path.join(save_dir,self.imageNames[-1]))
        fig.clf()
        plt.close(fig)
    

    def compareImages(self,img_org, gtImg, predImg):
        tp = 0
        fp = 0
        fn = 0
        
        GtDict, cellCountGT, immuneCountGt, tumorCountGT = self._findObjects(gtImg)
        predDict, cellCountPred, immuneCountPred, tumorCountPred = self._findObjects(predImg)
        
        cellAccuracy = round(abs(cellCountGT-cellCountPred)/(cellCountGT+self.smoothening_factor),4)
        immuneAccuracy = round(abs(immuneCountGt-immuneCountPred)/(immuneCountGt+self.smoothening_factor),4)
        tumorAccuracy = round(abs(tumorCountGT-tumorCountPred)/(tumorCountGT+self.smoothening_factor),4)

        self.cellCounts['GT'].append(cellCountGT)
        self.immuneCounts['GT'].append(immuneCountGt)
        self.tumorCounts['GT'].append(tumorCountGT)

        self.cellCounts['Pred'].append(cellCountPred)
        self.immuneCounts['Pred'].append(immuneCountPred)
        self.tumorCounts['Pred'].append(tumorCountPred)

        self.cellCounts['Accuracy'].append(cellAccuracy)
        self.immuneCounts['Accuracy'].append(immuneAccuracy)
        self.tumorCounts['Accuracy'].append(tumorAccuracy)

        ratioImmoGT = immuneCountGt/(immuneCountGt+tumorCountGT+cellCountGT+self.smoothening_factor)
        ratioImmoPred = immuneCountPred/(immuneCountPred+tumorCountPred+cellCountPred+self.smoothening_factor)
        ratioImmoAccuracy = round(abs(ratioImmoGT-ratioImmoPred),4)
        
        self.ratio['GTImmo'].append(ratioImmoGT)
        self.ratio['PredImmo'].append(ratioImmoPred)
        self.ratio['AccuracyImmo'].append(ratioImmoAccuracy)

        ratioImmoTummorGT = immuneCountGt/(immuneCountGt+tumorCountGT+self.smoothening_factor)
        ratioImmoTummorPred = immuneCountPred/(immuneCountPred+tumorCountPred+self.smoothening_factor)
        ratioImmoTummorAccuracy = round(abs(ratioImmoTummorGT-ratioImmoTummorPred),4)        
        self.ratio['GTImmoTummor'].append(ratioImmoTummorGT)
        self.ratio['PredImmoTummor'].append(ratioImmoTummorPred)
        self.ratio['AccuracyImmoTummor'].append(ratioImmoTummorAccuracy)

        currentClassDict = {}
        for i in self.classDict:
            currentClassDict[i] = {'tp':0,'TotalGT':0,'TotalPred':0}
        for gt in GtDict:
            (xGt,yGt),radiusGT = cv2.minEnclosingCircle(GtDict[gt]['contour'])
            currentClassDict[GtDict[gt]['class']]['TotalGT'] += 1
            for pred in predDict:
                (xPred,yPred),radiusPred = cv2.minEnclosingCircle(predDict[pred]['contour'])
                currentED = distance.euclidean((xGt,yGt),(xPred, yPred))
                if currentED< 5 and predDict[pred]['class']==GtDict[gt]['class']:
                    tp +=1
                    currentClassDict[predDict[pred]['class']]['tp'] +=1
                    self.edList.append(currentED)
                    continue

        fp = len(predDict) - tp
        fn = len(GtDict) - tp
        self.tp += tp
        self.fp += fp
        self.fn += fn
        
        for pred in predDict:
            currentClassDict[predDict[pred]['class']]['TotalPred'] += 1
            
        for i in self.classDict:
            self.classRes[i]['tp'] += currentClassDict[i]['tp']
            self.classRes[i]['fp'] += (currentClassDict[i]['TotalPred'] - currentClassDict[i]['tp'])
            self.classRes[i]['fn'] += (currentClassDict[i]['TotalGT'] - currentClassDict[i]['tp'])


        self.recall.append(round(tp / len(GtDict), 4))
        self.precision.append(round(tp / (tp + fp), 4))
        self.f1.append(round(tp / (tp + (0.5 * (fp + fn))), 4))
        self._save_visuals(img_org, gtImg, predImg, (tumorCountGT, immuneCountGt), (tumorCountPred, immuneCountPred), self.save_dir)
    def save(self):
        performace_results = {'sample name':self.imageNames,'precision':self.precision,
                              'recall':self.recall, 'f1':self.f1, 
                                'cell count Gold':self.cellCounts['GT'],
                              'cell count Pred': self.cellCounts['Pred'],
                              'cell count accuracy': self.cellCounts['Accuracy'],
                              'immune count Gold':self.immuneCounts['GT'],
                            'immune count Pred':self.immuneCounts['Pred'],
                            'immune accuracy':self.immuneCounts['Accuracy'],
                                'tumor count Gold':self.tumorCounts['GT'],
                              'tumor count Pred': self.tumorCounts['Pred'],
                              'tumor count accuracy': self.tumorCounts['Accuracy'],
                            'ratio Gold - 1':self.ratio['GTImmo'],
                            'ratio Pred - 1':self.ratio['PredImmo'],
                            'ratio accuracy - 1':self.ratio['AccuracyImmo'],
                            'ratio Gold - 2':self.ratio['GTImmoTummor'],
                            'ratio Pred -2':self.ratio['PredImmoTummor'],
                            'ratio accuracy - 2':self.ratio['AccuracyImmoTummor']}

        perf_dt = pd.DataFrame(performace_results)
        perf_dt.to_csv(os.path.join(self.save_dir,'resultsData.csv'), index=False)
        
        
        
        precision = self.tp/(self.tp+self.fp)
        recall = self.tp/(self.tp+self.fn)
        f1score = 2*precision*recall/(precision+recall)
        meanPrecision = sum(self.precision)/len(self.precision)
        meanRecall = sum(self.recall)/len(self.recall)
        meanf1 = sum(self.f1)/len(self.f1)
        meanED = sum(self.edList)/len(self.edList)
        sns_hist = sns.histplot(self.edList)
        fig = sns_hist.get_figure()
        fig.savefig(os.path.join(self.save_dir,'EDhist.png'))
        
        filteredCellAccuracy = [min(num, 5) for num in self.cellCounts['Accuracy']]
        filteredImmuneAccuracy = [min(num, 5) for num in self.immuneCounts['Accuracy']]
        filteredTumorAccuracy = [min(num, 5) for num in self.tumorCounts['Accuracy']]
        
        # filteredCellAccuracy = list(filter(lambda num: num < 5, self.cellCounts['Accuracy']))
        # filteredImmuneAccuracy = list(filter(lambda num: num < 5, self.immuneCounts['Accuracy']))
        # filteredTumorAccuracy = list(filter(lambda num: num < 5, self.tumorCounts['Accuracy']))

        classPrecision = []
        classRecall = []
        classF1 = []
        for cls in self.classRes:
            tp, fp, fn = self.classRes[cls]['tp'], self.classRes[cls]['fp'], self.classRes[cls]['fn']
            total_sample = tp + fn
            recall = round(tp / total_sample, 4)
            precision = round(tp / (tp + fp + self.smoothening_factor), 4)
            f_one = round((2*precision*recall)/(precision +
                                              recall + self.smoothening_factor), 4)
            classPrecision.append(precision)
            classRecall.append(recall)
            classF1.append(f_one)

        self.performace_results = {'precision':round(precision,4)*100,'recall':round(recall,4)*100,
                              'f1':round(f1score,4)*100, 'mean Precision':round(meanPrecision,4)*100,
                              'mean Recall':round(meanRecall,4)*100, 'mean f1':round(meanf1,4)*100,
                              'mean IoU':round(meanED,2), 
                              'Cell Precesion': classPrecision[0]*100, 
                              'Cell Recall': classRecall[0]*100,
                              'Cell F1': classF1[0]*100,
                              'Cell Accuracy': round(sum(filteredCellAccuracy)/len(filteredCellAccuracy),4),
                              'Immune Precesion': classPrecision[1]*100,
                              'Immune Recall': classRecall[1]*100,
                              'Immune F1': classF1[1]*100,
                              'Immune Accuracy': round(sum(filteredImmuneAccuracy)/len(filteredImmuneAccuracy),4),
                                'Tumor Precesion': classPrecision[2]*100,
                              'Tumor Recall': classRecall[2]*100,
                              'Tumor F1': classF1[2]*100,
                              'Tumor Accuracy': round(sum(filteredTumorAccuracy)/len(filteredTumorAccuracy),4),
                              'Ratio Accuracy-1': round(sum(self.ratio['AccuracyImmo'])/len(self.ratio['AccuracyImmo']),4),
                              'Ratio Accuracy-2': round(sum(self.ratio['AccuracyImmoTummor'])/len(self.ratio['AccuracyImmoTummor']),4)}
        
        perf_dt2 = pd.DataFrame(self.performace_results, index=[0])
        perf_dt2.to_csv(os.path.join(self.save_dir,'results.csv'), index=False)
        
    def getResults(self):
        return self.performace_results

class Results2Class:
    def __init__(self, save_dir, save_image = True):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.recall = []
        self.precision = []
        self.f1 = []
        self.classDict =  {1:'other',
                            2:'immune'}
        self.performace_results = {}
        self.cellCounts = {'GT':[],'Pred':[], 'AbsDiff':[], 'Accuracy':[],
                           'AccuracyRelative':[],'AccuracyRelativePD':[]}
        self.immuneCounts = {'GT':[],'Pred':[], 'AbsDiff':[], 'Accuracy':[],
                           'AccuracyRelative':[],'AccuracyRelativePD':[]}
        self.ratio = {'GT':[],'Pred':[], 'AbsDiff':[], 'Accuracy':[],
                           'AccuracyRelative':[],'AccuracyRelativePD':[]}
        self.classRes = {}
        self.imageNames = []
        self.G1metrics = []
        self.G2metrics = []
        self.G3metrics = []
        self.label_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in self.classDict:
            self.classRes[i] = {'tp':0,'fp':0,'fn':0,'tn':0}
        self.save_dir = save_dir
        self.sigma_list=[5, 20]
        self.sigma_thresh_list=list(np.arange(0.5,1, 0.05))
        self.arr_prec_immune=np.zeros((len(self.sigma_list), len(self.sigma_thresh_list)))
        self.arr_recall_immune=np.zeros((len(self.sigma_list), len(self.sigma_thresh_list)))
        self.arr_f1_immune=np.zeros((len(self.sigma_list), len(self.sigma_thresh_list)))
        self.arr_prec_other=np.zeros((len(self.sigma_list), len(self.sigma_thresh_list)))
        self.arr_recall_other=np.zeros((len(self.sigma_list), len(self.sigma_thresh_list)))        
        self.arr_f1_other=np.zeros((len(self.sigma_list), len(self.sigma_thresh_list)))
        self.save_image = save_image

    def _findObjects(self, img):
        objectDict = {}
        #create a dict to keep number of each class
        cellCounts = {}
        for cls in self.classDict:
            cellCounts[cls] = 0
        
        #for each class
        for cls in self.classDict:
            #create a binary image
            imgCls = np.zeros_like(img)
            imgCls[img==cls] = 1
            # find counter
            contours, _ = cv2.findContours(imgCls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #keep its total number
            cellCounts[cls] = len(contours)
            #iterate over counters
            e_coord_y = []
            e_coord_x = []
            for idx in range(len(contours)):
                #extract its center coordinates
                contour_i = contours[idx]
                M = cv2.moments(contour_i)
                #print(M)
                if(M['m00'] == 0):
                    continue;
                cx = round(M['m10'] / M['m00'])
                cy = round(M['m01'] / M['m00'])
                # keep center coordinates with its
                e_coord_y.append(cy)
                e_coord_x.append(cx)
            objectDict[cls] = (np.array(e_coord_x), np.array(e_coord_y))

        return objectDict, cellCounts[1], cellCounts[2]


    def _create_rgb_mask(self, mask):
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[mask == 1] = self.label_colors[0]
        rgb_mask[mask == 2] = self.label_colors[1]
        rgb_mask[mask == 3] = self.label_colors[2]

        return rgb_mask
    
    
    def _save_visuals(self, img_org, mask_img, prediction, cellCountsGT, cellCountsPred, save_dir):
        cellCountsGtOthers, cellCountsGtImmune = cellCountsGT
        cellCountsPredOther, cellCountsPredImmune = cellCountsPred
        
        fig, axs = plt.subplots(1, 3)
        fig.set_figheight(12)
        fig.set_figwidth(30)
        if len(img_org.shape) == 3:
            axs[0].imshow(img_org)
            axs[0].title.set_text('image')
        else:
            axs[0].imshow(img_org, cmap='gray')
            axs[0].title.set_text('image')
        axs[1].imshow(self._create_rgb_mask(mask_img))
        axs[1].title.set_text('label')
        fig.text(.51, .17, "tumor: {}".format(cellCountsGtOthers), ha='center',color="red")
        fig.text(.51, .15, "immune {}".format(cellCountsGtImmune), ha='center',color="green")
        axs[2].imshow(self._create_rgb_mask(prediction))
        axs[2].title.set_text('prediction')
        fig.text(.79, .17, "tumor: {}".format(cellCountsPredOther), ha='center',color="red")
        fig.text(.79, .15, "immune {}".format(cellCountsPredImmune), ha='center',color="green")
        fig.savefig(os.path.join(save_dir,self.imageNames[-1]))
        fig.clf()
        plt.close(fig)
        

    def compareImages(self, img_org, gtImg, predImg, tsv_path):

        # create dot image from ground-truth csv 
        gt_dot_other, gt_dot_immune = create_label_coordinates(tsv_path)
        # calculate gold counts
        cellCountGt = np.sum(gt_dot_other)
        immuneCountGt = np.sum(gt_dot_immune)

        # compute estimated counts as well as cell location
        predDict, cellCountPred, immuneCountPred = self._findObjects(predImg)
        
        # compute cell and immune counting success metrics
        abs_diff_cell, cellAccuracy, cellAccuracyRelative, cellAccuracyRelativePD = countAccuracyMetric(cellCountGt, cellCountPred)
        abs_diff_immune, immuneAccuracy, immuneAccuracyRelative, immuneAccuracyRelativePD = countAccuracyMetric(immuneCountGt, immuneCountPred)
        
        # compute ratio and ratio estimation success metrics
        ratioGT = immuneCountGt/(cellCountGt+immuneCountGt)
        ratioPred = immuneCountPred/(cellCountPred+immuneCountPred)
        abs_diff_ratio, ratioAccuracy, ratioAccuracyRelative, ratioAccuracyRelativePD = countAccuracyMetric(ratioGT, ratioPred)

        # keep metrics for the image
        self.cellCounts['GT'].append(cellCountGt)
        self.cellCounts['Pred'].append(cellCountPred)
        self.cellCounts['AbsDiff'].append(abs_diff_cell)
        self.cellCounts['Accuracy'].append(cellAccuracy)
        self.cellCounts['AccuracyRelative'].append(cellAccuracyRelative)
        self.cellCounts['AccuracyRelativePD'].append(cellAccuracyRelativePD)

        self.immuneCounts['GT'].append(immuneCountGt)
        self.immuneCounts['Pred'].append(immuneCountPred)
        self.immuneCounts['AbsDiff'].append(abs_diff_immune)
        self.immuneCounts['Accuracy'].append(immuneAccuracy)
        self.immuneCounts['AccuracyRelative'].append(immuneAccuracyRelative)
        self.immuneCounts['AccuracyRelativePD'].append(immuneAccuracyRelativePD)
        
        self.ratio['GT'].append(ratioGT)
        self.ratio['Pred'].append(ratioPred)
        self.ratio['AbsDiff'].append(round(abs_diff_ratio,4))
        self.ratio['Accuracy'].append(ratioAccuracy)
        self.ratio['AccuracyRelative'].append(ratioAccuracyRelative)
        self.ratio['AccuracyRelativePD'].append(ratioAccuracyRelativePD)
        
        e_dot_other = np.zeros_like(gt_dot_other)
        e_coord_other = predDict[1]
        e_coord_x, e_coord_y = e_coord_other
        for e_indx in range(len(e_coord_y)):
            e_dot_other[e_coord_y[e_indx], e_coord_x[e_indx]] = 1
            
        e_dot_immune = np.zeros_like(gt_dot_immune)
        e_coord_immune = predDict[2]
        e_coord_x, e_coord_y = e_coord_immune
        for e_indx in range(len(e_coord_y)):
            e_dot_immune[e_coord_y[e_indx], e_coord_x[e_indx]] = 1
          
        gmae_cell_res = GMAE(1, gt_dot_other, e_dot_other)
        gmae_immune_res = GMAE(1, gt_dot_immune, e_dot_immune)
        self.G1metrics.append(gmae_cell_res+gmae_immune_res)
        
        gmae_cell_res = GMAE(2, gt_dot_other, e_dot_other)
        gmae_immune_res = GMAE(2, gt_dot_immune, e_dot_immune)
        self.G2metrics.append(gmae_cell_res+gmae_immune_res)
        
        gmae_cell_res = GMAE(3, gt_dot_other, e_dot_other)
        gmae_immune_res = GMAE(3, gt_dot_immune, e_dot_immune)
        self.G3metrics.append(gmae_cell_res+gmae_immune_res)

        arr_prec_current, arr_recall_current, arr_f1_current = CrowdMatchingTest(gt_dot_immune, 
                                                                                 predDict[2], 
                                                                                 self.sigma_list,
                                                                                 self.sigma_thresh_list,
                                                                                 inputType='Coordinates')
        self.arr_prec_immune += arr_prec_current
        self.arr_recall_immune += arr_recall_current
        self.arr_f1_immune += arr_f1_current
        
        arr_prec_current, arr_recall_current, arr_f1_current = CrowdMatchingTest(gt_dot_other, 
                                                                                 predDict[1], 
                                                                                 self.sigma_list,
                                                                                 self.sigma_thresh_list,
                                                                                 inputType='Coordinates')
        
        self.arr_prec_other += arr_prec_current
        self.arr_recall_other += arr_recall_current     
        self.arr_f1_other += arr_f1_current
        
        if self.save_image:
            self._save_visuals(img_org, gtImg, predImg, (cellCountGt, immuneCountGt), (cellCountPred, immuneCountPred), self.save_dir)


    def save(self):
        performace_results = {'sample name':self.imageNames,
                              'cell count Gold':self.cellCounts['GT'],
                              'cell count Pred': self.cellCounts['Pred'],
                            'cell abs diff': self.cellCounts['AbsDiff'],
                              'cell accuracy': self.cellCounts['Accuracy'],
                            'cell accuracy RD':self.cellCounts['AccuracyRelative'],
                            'cell accuracy RD Perantage':self.cellCounts['AccuracyRelativePD'],     
                              
                              'immune count Gold':self.immuneCounts['GT'],
                            'immune count Pred':self.immuneCounts['Pred'],
                            'immune abs diff': self.immuneCounts['AbsDiff'],
                            'immune accuracy':self.immuneCounts['Accuracy'],
                            'immune accuracy RD':self.immuneCounts['AccuracyRelative'],
                            'immune accuracy RD Perantage':self.immuneCounts['AccuracyRelativePD'],                            
                            'ratio Gold':self.ratio['GT'],
                            'ratio Pred':self.ratio['Pred'],
                            'ratio abs diff': self.ratio['AbsDiff'],
                            'ratio accuracy':self.ratio['Accuracy'],
                            'ratio accuracy RD':self.ratio['AccuracyRelative'],
                            'ratio accuracy RD Perantage':self.ratio['AccuracyRelativePD']}
        perf_dt = pd.DataFrame(performace_results)
        perf_dt.to_csv(os.path.join(self.save_dir,'resultsData.csv'), index=False)
        

        # Column names (corresponding to your returned values)
        columns = [
            "gmae_cell",
            "gmae_cellAccuracyRelative",
            "gmae_cellAccuracyRelativePD",
            "gmae_immune",
            "gmae_immuneAccuracyRelative",
            "gmae_immuneAccuracyRelativePD",
        ]

        # Convert to DataFrame
        df1 = pd.DataFrame(self.G1metrics, columns=columns)
        df2 = pd.DataFrame(self.G2metrics, columns=columns)
        df3 = pd.DataFrame(self.G3metrics, columns=columns)
        
        column_means1 = df1.mean().to_numpy()  # Or use .values for older pandas versions
        column_means2 = df2.mean().to_numpy()  # Or use .values for older pandas versions
        column_means3 = df3.mean().to_numpy()  # Or use .values for older pandas versions

        data = [column_means1, column_means2, column_means3]
        res = pd.DataFrame(data, columns=columns, index=["G(1)", "G(2)", "G(3)"])
        res.to_csv(os.path.join(self.save_dir,'resultsGridCount.csv'), index=True)
        
        self.arr_f1_immune /= len(self.imageNames)
        self.arr_prec_immune /= len(self.imageNames)
        self.arr_recall_immune /= len(self.imageNames)
        
        self.arr_f1_other /= len(self.imageNames)
        self.arr_prec_other /= len(self.imageNames)
        self.arr_recall_other /= len(self.imageNames)
        
        columns = [
            "prec_cell",
            "recall_cell",
            "f1_cell",
            "prec_immune",
            "recall_immune",
            "f1_immune"
        ]
        
        index=["sigma(5)", "sigma(20)", "sigma(5)_09", "sigma(20)_09"]
        
        sigma5prec_immune, sigma20prec_immune = np.mean(self.arr_prec_immune, axis=1)
        sigma5recall_immune, sigma20recall_immune = np.mean(self.arr_recall_immune, axis=1)
        sigma5f1_immune, sigma20f1_immune = np.mean(self.arr_f1_immune, axis=1)
        
        sigma5prec_other, sigma20prec_other = np.mean(self.arr_prec_other, axis=1)
        sigma5recall_other, sigma20recall_other = np.mean(self.arr_recall_other, axis=1)
        sigma5f1_other, sigma20f1_other = np.mean(self.arr_f1_other, axis=1)

        
        row1 = [sigma5prec_other, sigma5recall_other, sigma5f1_other , sigma5prec_immune, sigma5recall_immune, sigma5f1_immune]
        row2 = [sigma20prec_other, sigma20recall_other, sigma20f1_other, sigma20prec_immune, sigma20recall_immune, sigma20f1_immune]
        
        sigma5prec_immune, sigma20prec_immune = np.mean(self.arr_prec_immune[:,:-1], axis=1)
        sigma5recall_immune, sigma20recall_immune = np.mean(self.arr_recall_immune[:,:-1], axis=1)
        sigma5f1_immune, sigma20f1_immune = np.mean(self.arr_f1_immune[:,:-1], axis=1)

        sigma5prec_other, sigma20prec_other = np.mean(self.arr_prec_other[:,:-1], axis=1)
        sigma5recall_other, sigma20recall_other = np.mean(self.arr_recall_other[:,:-1], axis=1)
        sigma5f1_other, sigma20f1_other = np.mean(self.arr_f1_other[:,:-1], axis=1)

        row3 = [sigma5prec_other, sigma5recall_other, sigma5f1_other, sigma5prec_immune, sigma5recall_immune, sigma5f1_immune]
        row4 = [sigma20prec_other, sigma20recall_other, sigma20f1_other, sigma20prec_immune, sigma20recall_immune, sigma20f1_immune]    
    
        # Create DataFrame
        res = pd.DataFrame([row1, row2, row3, row4], columns=columns, index=index)

        # Save DataFrame as CSV
        res.to_csv(os.path.join(self.save_dir,'resultsMatching.csv'), index=True)
        
        gt_list_immune = self.immuneCounts['GT']
        pred_list_immune = self.immuneCounts['Pred']
        
        gt_list_other = self.cellCounts['GT'] 
        pred_list_other = self.cellCounts['Pred']

        plt.scatter(gt_list_immune, pred_list_immune, c ="black")
        plt.xlabel('golds')
        plt.ylabel('predictions')
        maxLimit = int(max(max(gt_list_immune), max(pred_list_immune))) + 100
        plt.xlim(0,maxLimit)
        plt.ylim(0,maxLimit)
        plt.plot(range(maxLimit))
        plt.savefig(os.path.join(self.save_dir,'resultsData_immune.png'))
        plt.cla()
    
        plt.scatter(gt_list_immune, pred_list_immune, c ="black")
        plt.xlabel('golds')
        plt.ylabel('predictions')
        plt.xlim(0,200)
        plt.ylim(0,200)
        plt.plot(range(maxLimit))
        plt.savefig(os.path.join(self.save_dir,'resultsData_immune_200.png'))
        plt.cla()
    
        plt.scatter(gt_list_immune, pred_list_immune, c ="black")
        plt.xlabel('golds')
        plt.ylabel('predictions')
        plt.xlim(0,50)
        plt.ylim(0,50)
        plt.plot(range(maxLimit))
        plt.savefig(os.path.join(self.save_dir,'resultsData_immune_50.png'))
        plt.cla()
    
        plt.scatter(gt_list_other, pred_list_other, c ="black")
        plt.xlabel('golds')
        plt.ylabel('predictions')
        maxLimit = int(max(max(gt_list_other), max(pred_list_other))) + 100
        plt.xlim(0,maxLimit)
        plt.ylim(0,maxLimit)
        plt.plot(range(maxLimit))
        plt.savefig(os.path.join(self.save_dir,'resultsData_other.png'))
        plt.cla()
        
        pearson_r_cell, _ = pearsonr(self.cellCounts['GT'], self.cellCounts['Pred'])        
        pearson_r_immune, _ = pearsonr(self.immuneCounts['GT'], self.immuneCounts['Pred'])
        pearson_r_ratio, _ = pearsonr(self.ratio['GT'], self.ratio['Pred'])
               
        filteredCellAccuracy = [min(num, 5) for num in self.cellCounts['Accuracy']]
        filteredImmuneAccuracy = [min(num, 5) for num in self.immuneCounts['Accuracy']]

        self.performace_results = {
                            'Cell MAE': round(sum( self.cellCounts['AbsDiff'])/len( self.cellCounts['AbsDiff']),4),
                              'Cell MRE': round(sum(filteredCellAccuracy)/len(filteredCellAccuracy),4),
                            'Cell MRE max': round(sum( self.cellCounts['AccuracyRelative'])/len( self.cellCounts['AccuracyRelative']),4),
                            'Cell RPD': round(sum( self.cellCounts['AccuracyRelativePD'])/len( self.cellCounts['AccuracyRelativePD']),4),
                            'Cell Pearson r': pearson_r_cell,
                            'Immune MAE': round(sum( self.immuneCounts['AbsDiff'])/len( self.immuneCounts['AbsDiff']),4),
                              'Immune MRE': round(sum(filteredImmuneAccuracy)/len(filteredImmuneAccuracy),4),
                            'Immune MRE max': round(sum( self.immuneCounts['AccuracyRelative'])/len( self.immuneCounts['AccuracyRelative']),4),
                            'Immune RPD': round(sum( self.immuneCounts['AccuracyRelativePD'])/len( self.immuneCounts['AccuracyRelativePD']),4),
                            'Immune Pearson r': pearson_r_immune,
                                'Ratio MAE': round(sum(self.ratio['AbsDiff'])/len( self.immuneCounts['AbsDiff']),4),
                              'Ratio MRE': round(sum(self.ratio['Accuracy'])/len(self.ratio['Accuracy']),4),
                            'Ratio Accuracy MRE max': round(sum(self.ratio['AccuracyRelative'])/len(self.ratio['AccuracyRelative']),4),
                            'Ratio Accuracy RPD': round(sum(self.ratio['AccuracyRelativePD'])/len(self.ratio['AccuracyRelativePD']),4),
                            'Ratio pearson r':pearson_r_ratio}
        
        
        perf_dt2 = pd.DataFrame(self.performace_results, index=[0])
        perf_dt2.to_csv(os.path.join(self.save_dir,'results.csv'), index=False)
        
        self.performace_results = {
                    'Cell MAE': round(sum( self.cellCounts['AbsDiff'])/len( self.cellCounts['AbsDiff']),4),
                        'Cell MRE': round(sum(filteredCellAccuracy)/len(filteredCellAccuracy),4),
                    'Cell MRE max': round(sum( self.cellCounts['AccuracyRelative'])/len( self.cellCounts['AccuracyRelative']),4),
                    'Cell RPD': round(sum( self.cellCounts['AccuracyRelativePD'])/len( self.cellCounts['AccuracyRelativePD']),4),
                    'Cell Pearson r': pearson_r_cell,
                    'Immune MAE': round(sum( self.immuneCounts['AbsDiff'])/len( self.immuneCounts['AbsDiff']),4),
                        'Immune MRE': round(sum(filteredImmuneAccuracy)/len(filteredImmuneAccuracy),4),
                    'Immune MRE max': round(sum( self.immuneCounts['AccuracyRelative'])/len( self.immuneCounts['AccuracyRelative']),4),
                    'Immune RPD': round(sum( self.immuneCounts['AccuracyRelativePD'])/len( self.immuneCounts['AccuracyRelativePD']),4), 
                    'Immune Pearson r': pearson_r_immune,    
                        'Ratio MAE': round(sum(self.ratio['AbsDiff'])/len( self.immuneCounts['AbsDiff']),4),
                        'Ratio MRE': round(sum(self.ratio['Accuracy'])/len(self.ratio['Accuracy']),4),
                    'Ratio MRE max': round(sum(self.ratio['AccuracyRelative'])/len(self.ratio['AccuracyRelative']),4),
                    'Ratio RPD': round(sum(self.ratio['AccuracyRelativePD'])/len(self.ratio['AccuracyRelativePD']),4),
                    'Ratio pearson r':pearson_r_ratio
                    }
        
        perf_dt2 = pd.DataFrame(self.performace_results, index=[0])
        perf_dt2.to_csv(os.path.join(self.save_dir,'resultsC.csv'), index=False)
        #### filter below 25 #####
        
        
        #### filter below 25 #####
        imageNamesFiltered = []
        cellCountsFiltered = []
        cellCountsFilteredGT = []
        cellCountsFilteredPred = []
        cellCountsFilteredAccuracy = []
        cellCountsFilteredAbsDiff = []
        cellCountsFilteredRD = []
        cellCountsFilteredRDP = []
        
        immuneCountsFilteredGT = []
        immuneCountsFilteredPred = []
        immuneCountsFilteredAccuracy = []
        immuneCountsFilteredAbsDiff = []
        immuneCountsFilteredRD = []
        immuneCountsFilteredRDP = []
        
        ratioFilteredGT = []
        ratioFilteredPred = []
        ratioFilteredAccuracy = []
        ratioFilteredAbsDiff = []
        ratioFilteredRD = []
        ratioFilteredRDP = []


        for i, immuneNum in enumerate(self.immuneCounts['GT']):
            if immuneNum<=25 or self.immuneCounts['Pred'][i]<25:
                continue
            imageNamesFiltered.append(self.imageNames[i])
            cellCountsFilteredGT.append(self.cellCounts['GT'][i])
            cellCountsFilteredPred.append(self.cellCounts['Pred'][i])
            cellCountsFilteredAccuracy.append(self.cellCounts['Accuracy'][i])
            cellCountsFilteredAbsDiff.append(self.cellCounts['AbsDiff'][i])
            cellCountsFilteredRD.append(self.cellCounts['AccuracyRelative'][i])
            cellCountsFilteredRDP.append(self.cellCounts['AccuracyRelativePD'][i])
            
            immuneCountsFilteredGT.append(self.immuneCounts['GT'][i])
            immuneCountsFilteredPred.append(self.immuneCounts['Pred'][i])
            immuneCountsFilteredAccuracy.append(self.immuneCounts['Accuracy'][i])
            immuneCountsFilteredAbsDiff.append(self.immuneCounts['AbsDiff'][i])
            immuneCountsFilteredRD.append(self.immuneCounts['AccuracyRelative'][i])
            immuneCountsFilteredRDP.append(self.immuneCounts['AccuracyRelativePD'][i])
            
            ratioFilteredGT.append(self.ratio['GT'][i])
            ratioFilteredPred.append(self.ratio['Pred'][i])
            ratioFilteredAccuracy.append(self.ratio['Accuracy'][i])
            ratioFilteredAbsDiff.append(self.ratio['GT'][i])
            ratioFilteredRD.append(self.ratio['AccuracyRelative'][i])
            ratioFilteredRDP.append(self.ratio['AccuracyRelativePD'][i])
            
        performace_results_filtered = {'sample name':imageNamesFiltered, 'cell count Gold':cellCountsFilteredGT,
                              'cell count Pred':cellCountsFilteredPred,
                            'cell abs diff': cellCountsFilteredAbsDiff,
                              'cell accuracy': cellCountsFilteredAccuracy,
                            'cell accuracy RD':cellCountsFilteredRD,
                            'cell accuracy RD Perantage':cellCountsFilteredRDP,     
                              
                              'immune count Gold':immuneCountsFilteredGT,
                            'immune count Pred':immuneCountsFilteredPred,
                            'immune abs diff': immuneCountsFilteredAbsDiff,
                            'immune accuracy': immuneCountsFilteredAccuracy,
                            'immune accuracy RD': immuneCountsFilteredRD,
                            'immune accuracy RD Perantage':immuneCountsFilteredRDP,        
                                                
                            'ratio Gold':ratioFilteredGT,
                            'ratio Pred':ratioFilteredPred,
                            'ratio abs diff': ratioFilteredAbsDiff,
                            'ratio accuracy': ratioFilteredAccuracy,
                            'ratio accuracy RD': ratioFilteredRD,
                            'ratio accuracy RD Perantage': ratioFilteredRDP}
        perf_dt_filtered = pd.DataFrame(performace_results_filtered)
        perf_dt_filtered.to_csv(os.path.join(self.save_dir,'resultsDataFiltred.csv'), index=False)
        
        performace_results_filtered = {
                    'Cell MAE': round(sum(cellCountsFilteredAbsDiff)/len(cellCountsFilteredAbsDiff),4),
                        'Cell Accuracy': round(sum(cellCountsFilteredAccuracy)/len(cellCountsFilteredAccuracy),4),
                    'Cell Accuracy RD': round(sum(cellCountsFilteredRD)/len(cellCountsFilteredRD),4),
                    'Cell Accuracy RPD': round(sum(cellCountsFilteredRDP)/len(cellCountsFilteredRDP),4),

                    'Immune MAE': round(sum(immuneCountsFilteredAbsDiff)/len(immuneCountsFilteredAbsDiff),4),
                        'Immune Accuracy': round(sum(immuneCountsFilteredAccuracy)/len(immuneCountsFilteredAccuracy),4),
                    'Immune Accuracy RD': round(sum(immuneCountsFilteredRD)/len(immuneCountsFilteredRD),4),
                    'Immune Accuracy RPD': round(sum(immuneCountsFilteredRDP)/len(immuneCountsFilteredRDP),4),
                    
                        'Ratio MAE': round(sum(ratioFilteredAbsDiff)/len(ratioFilteredAbsDiff),4),
                        'Ratio Accuracy': round(sum(ratioFilteredAccuracy)/len(ratioFilteredAccuracy),4),
                    'Ratio Accuracy RD': round(sum(ratioFilteredRD)/len(ratioFilteredRD),4),
                    'Ratio Accuracy RPD': round(sum(ratioFilteredRDP)/len(ratioFilteredRDP),4)}
        
        perf_dt2 = pd.DataFrame(performace_results_filtered, index=[0])
        perf_dt2.to_csv(os.path.join(self.save_dir,'resultsFiltered.csv'), index=False)
        
    def getResults(self):
        return self.performace_results


def test_single_mc(model, device, input_size, ch, Num_Class, image_list, tsv_files, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if Num_Class == 3:
        res = Results2Class(save_dir, False)
    elif Num_Class == 4:
        res = Results3Class(save_dir)
    else:
        print('invalid')
        quit()
    
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        img_org = cv2.imread(img_path)
        
        # im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
        # img_org = NORMALIZER.transform(im_rgb)
        
        # ihc_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
        # rihc_hed = rgb2hed(ihc_rgb)
        # img_org = rihc_hed[:,:,0]

        img_input = preprocess(img_org, input_size)
        imgHeight, imgWidth = img_org.shape[:2]
        model.to(device=device)
        model.eval()
        with torch.no_grad():
            outputs = model(img_input)
            probs = F.softmax(outputs, dim=1)
            out = torch.argmax(probs, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if imgHeight != input_size[0] or imgWidth != input_size[1]:
                pred = zoom(out, (imgHeight / input_size[0], imgWidth / input_size[1]), order=0)
            else:
                pred = out
        pred = np.uint8(pred)

        #cv2.imwrite(os.path.join(save_dir, image_name+'_pred.png'),  pred)


        # read gt mask
        mask_path = img_path[:img_path.rfind('.')] + '_label_mc.png'
        mask = cv2.imread(mask_path, 0)
        
        img_name = img_path.split('/')[-1].split('.png')[0]
        tsv_path = tsv_files[img_name]
        
        res.imageNames.append(image_name)

        res.compareImages(img_org, mask, pred, tsv_path)
    res.save()    
    resDict = res.getResults()
    return resDict


def test_single_reg(model, device, input_size, ch, Num_Class, image_list, tsv_files, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    sigma_list=[5, 20]
    sigma_thresh_list=list(np.arange(0.5,1, 0.05))

    sample_list = []
    
    gt_list_immune = []
    pred_list_immune = []
    Absdiff_list_immune = []
    accuracy_immune = []
    AccuracyRelative_immune = []
    AccuracyRelativePD_immune = []
    
    gt_list_other = []
    pred_list_other = []
    Absdiff_list_other = []
    accuracy_other = []
    AccuracyRelative_other = []
    AccuracyRelativePD_other = []    
    
    gt_list_ratio = []
    pred_list_ratio = []
    Absdiff_list_ratio = []
    accuracy_ratio = []
    AccuracyRelative_ratio = []
    AccuracyRelativePD_ratio = []  
        
    smoothening_factor = 1e-7
    G1metrics = []
    G2metrics = []
    G3metrics = []
    
    arr_prec_immune=np.zeros((len(sigma_list), len(sigma_thresh_list)))
    arr_recall_immune=np.zeros((len(sigma_list), len(sigma_thresh_list)))
    arr_f1_immune=np.zeros((len(sigma_list), len(sigma_thresh_list)))
    arr_prec_other=np.zeros((len(sigma_list), len(sigma_thresh_list)))
    arr_recall_other=np.zeros((len(sigma_list), len(sigma_thresh_list)))        
    arr_f1_other=np.zeros((len(sigma_list), len(sigma_thresh_list)))
       
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        sample_list.append(image_name)
        img_org = cv2.imread(img_path)
        
        # im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
        # img_org = NORMALIZER.transform(im_rgb)
        # ihc_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
        # rihc_hed = rgb2hed(ihc_rgb)
        # img_org = rihc_hed[:,:,0]

        img_input = preprocess(img_org, input_size)
        imgHeight, imgWidth = img_org.shape[:2]
        model.to(device=device)
        model.eval()
        with torch.no_grad():
            output = torch.nn.functional.relu(model(img_input))
            out = output.squeeze(0).cpu().detach().numpy()

            out_immune = out[1,:,:]
            out_other = out[0,:,:]            
            if imgHeight != input_size[0] or imgWidth != input_size[1]:
                pred_other = zoom(out_other, (imgHeight / input_size[0], imgWidth / input_size[1]), order=0)
                pred_immune = zoom(out_immune, (imgHeight / input_size[0], imgWidth / input_size[1]), order=0)

            else:
                pred_other = out_other
                pred_immune = out_immune
        pred_other = pred_other/200
        pred_immune = pred_immune/200

        # Read tsv path and create dot map
        img_name = img_path.split('/')[-1].split('.png')[0]
        tsv_path = tsv_files[img_name]
        gt_dot_other, gt_dot_immune = create_label_coordinates(tsv_path)

        # read gt mask
        mask_path = img_path[:img_path.rfind('.')] + '_label_reg.npy'
        mask = np.load(mask_path)
        mask_other = mask[:,:,0]
        mask_immune = mask[:,:,1]
        
        ##### Count other cells #####
        cellCountGt_other = np.sum(gt_dot_other)
        cellCountPred_other = np.sum(pred_other)
        cellCountAbsDiff_other, otherAccuracy, otherAccuracyRelative, otherAccuracyRelativePD = countAccuracyMetric(cellCountGt_other, cellCountPred_other)
        
        ## add to list
        gt_list_other.append(round(cellCountGt_other, 4))
        pred_list_other.append(round(cellCountPred_other, 4))
        Absdiff_list_other.append(round(cellCountAbsDiff_other,4))
        accuracy_other.append(round(otherAccuracy, 4))
        AccuracyRelative_other.append(round(otherAccuracyRelative, 4))
        AccuracyRelativePD_other.append(round(otherAccuracyRelativePD, 4))
        
        ##### Count immune cells #####
        cellCountGt_immune = np.sum(gt_dot_immune)
        cellCountPred_immune = np.sum(pred_immune)
        ## calculate absolute difference
        cellCountAbsDiff_immune, immuneAccuracy, immuneAccuracyRelative, immuneAccuracyRelativePD = countAccuracyMetric(cellCountGt_immune, cellCountPred_immune)
        
        #add to list
        gt_list_immune.append(round(cellCountGt_immune,4))
        pred_list_immune.append(round(cellCountPred_immune,4))
        Absdiff_list_immune.append(round(cellCountAbsDiff_immune,4))
        accuracy_immune.append(round(immuneAccuracy, 4))        
        AccuracyRelative_immune.append(round(immuneAccuracyRelative, 4))
        AccuracyRelativePD_immune.append(round(immuneAccuracyRelativePD, 4))    
    
        # calculate GT and Pred ratio
        ratioGT = cellCountGt_immune/(cellCountGt_other+cellCountGt_immune)
        ratioPred = cellCountPred_immune/(cellCountPred_other+cellCountPred_immune)
        abs_diff_ratio, ratioAccuracy, ratioAccuracyRelative, ratioAccuracyRelativePD = countAccuracyMetric(ratioGT, ratioPred)
                
        gt_list_ratio.append(ratioGT)
        pred_list_ratio.append(ratioPred)
        Absdiff_list_ratio.append(abs_diff_ratio)
        accuracy_ratio.append(ratioAccuracy)
        AccuracyRelative_ratio.append(ratioAccuracyRelative)
        AccuracyRelativePD_ratio.append(ratioAccuracyRelativePD)
        
        
        
        if False:
            fig, axs = plt.subplots(2, 3)
            fig.set_figheight(20)
            fig.set_figwidth(30)
            if len(img_org.shape) == 3:
                axs[0, 0].imshow(img_org)
                axs[0, 0].title.set_text('image')
                axs[1, 0].imshow(img_org)
            else:
                axs[0, 0].imshow(img_org, cmap='gray')
                axs[0, 0].title.set_text('image')
                axs[1, 0].imshow(img_org, cmap='gray')

            axs[0, 1].imshow(mask_immune)
            axs[0, 1].title.set_text('label immune')
            fig.text(.51, .5, "immune count: {}".format(cellCountGt_immune), ha='center', fontsize = 16)
            axs[0, 2].imshow(pred_immune)
            axs[0, 2].title.set_text('prediction immune')
            fig.text(.75, .5, "immune count: {}".format(cellCountPred_immune), fontsize = 16)

            axs[1, 1].imshow(mask_other)
            axs[1, 1].title.set_text('label other')
            fig.text(.51, .08, "other count: {}".format(cellCountGt_other), ha='center', fontsize = 16)
            axs[1, 2].imshow(pred_other)
            axs[1, 2].title.set_text('prediction other')
            fig.text(.75, .08, "other count: {}".format(cellCountPred_other), fontsize = 16)
            
            fig.savefig(os.path.join(save_dir,image_name))
            fig.clf()
            plt.close(fig)
            image_name2 = image_name[:image_name.rfind('.')]
            # a colormap and a normalization 
            cmap = plt.cm.viridis

            plt.imsave(os.path.join(save_dir,image_name2+'_pred_other.png'), pred_other, cmap=cmap)
            plt.imsave(os.path.join(save_dir,image_name2+'_gt_other.png'), mask_other, cmap=cmap)
            plt.imsave(os.path.join(save_dir,image_name2+'_pred_immune.png'), pred_immune, cmap=cmap)
            plt.imsave(os.path.join(save_dir,image_name2+'_gt_immune.png'), mask_immune, cmap=cmap)
            
            
        gmae_cell_res = GMAE(1, gt_dot_other, pred_other)
        gmae_immune_res = GMAE(1, gt_dot_immune, pred_immune)
        G1metrics.append(gmae_cell_res+gmae_immune_res)
        
        gmae_cell_res = GMAE(2, gt_dot_other, pred_other)
        gmae_immune_res = GMAE(2, gt_dot_immune, pred_immune)
        G2metrics.append(gmae_cell_res+gmae_immune_res)
        
        gmae_cell_res = GMAE(3, gt_dot_other, pred_other)
        gmae_immune_res = GMAE(3, gt_dot_immune, pred_immune)
        G3metrics.append(gmae_cell_res+gmae_immune_res)
                
        arr_prec_current, arr_recall_current, arr_f1_current = CrowdMatchingTest(gt_dot_immune, 
                                                                                 pred_immune, 
                                                                                 sigma_list,
                                                                                 sigma_thresh_list,
                                                                                 inputType='Regression')
        arr_prec_immune += arr_prec_current
        arr_recall_immune += arr_recall_current
        arr_f1_immune += arr_f1_current
        
        arr_prec_current, arr_recall_current, arr_f1_current = CrowdMatchingTest(gt_dot_other, 
                                                                                 pred_other, 
                                                                                 sigma_list,
                                                                                 sigma_thresh_list,
                                                                                 inputType='Regression')
        
        arr_prec_other += arr_prec_current
        arr_recall_other += arr_recall_current
        arr_f1_other += arr_f1_current
         
    plt.scatter(gt_list_immune, pred_list_immune, c ="black")
    plt.xlabel('golds')
    plt.ylabel('predictions')
    maxLimit = int(max(max(gt_list_immune), max(pred_list_immune))) + 100
    plt.xlim(0,maxLimit)
    plt.ylim(0,maxLimit)
    plt.plot(range(maxLimit))
    plt.savefig(os.path.join(save_dir,'resultsData_immune.png'))
    plt.cla()
    
    plt.scatter(gt_list_immune, pred_list_immune, c ="black")
    plt.xlabel('golds')
    plt.ylabel('predictions')
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.plot(range(maxLimit))
    plt.savefig(os.path.join(save_dir,'resultsData_immune_200.png'))
    plt.cla()
    
    plt.scatter(gt_list_immune, pred_list_immune, c ="black")
    plt.xlabel('golds')
    plt.ylabel('predictions')
    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.plot(range(maxLimit))
    plt.savefig(os.path.join(save_dir,'resultsData_immune_50.png'))
    plt.cla()
    
    plt.scatter(gt_list_other, pred_list_other, c ="black")
    plt.xlabel('golds')
    plt.ylabel('predictions')
    maxLimit = int(max(max(gt_list_other), max(pred_list_other))) + 100
    plt.xlim(0,maxLimit)
    plt.ylim(0,maxLimit)
    plt.plot(range(maxLimit))
    plt.savefig(os.path.join(save_dir,'resultsData_other.png'))
    plt.cla()

    
    performace_results = {'sample name':sample_list,
                        'cell count Gold':gt_list_other,
                        'cell count Pred': pred_list_other,
                    'cell abs diff': Absdiff_list_other,
                        'cell accuracy': accuracy_other,
                    'cell accuracy RD': AccuracyRelative_other,
                    'cell accuracy RD Perantage':AccuracyRelativePD_other,     
                    'immune count Gold':gt_list_immune,
                    'immune count Pred':pred_list_immune,
                    'immune abs diff':Absdiff_list_immune,
                    'immune accuracy':accuracy_immune,
                    'immune accuracy RD':AccuracyRelative_immune,
                    'immune accuracy RD Perantage':AccuracyRelativePD_immune,                            
                    'ratio Gold':gt_list_ratio,
                    'ratio Pred':pred_list_ratio,
                    'ratio abs diff': Absdiff_list_ratio,
                    'ratio accuracy': accuracy_ratio,
                    'ratio accuracy RD': AccuracyRelative_ratio,
                    'ratio accuracy RD Perantage':AccuracyRelativePD_ratio}

    perf_dt = pd.DataFrame(performace_results)
    perf_dt.to_csv(os.path.join(save_dir,'resultsData.csv'), index=False)
    
    pearson_r_cell, _ = pearsonr(gt_list_other, pred_list_other)        
    pearson_r_immune, _ = pearsonr(gt_list_immune, pred_list_immune)
    pearson_r_ratio, _ = pearsonr(gt_list_ratio, pred_list_ratio)
        
            
    filteredCellAccuracy = [min(num, 5) for num in accuracy_other]
    filteredImmuneAccuracy = [min(num, 5) for num in accuracy_immune]
            
    filteredCellAccuracy_mean = round(sum(filteredCellAccuracy)/len(filteredCellAccuracy),4)
    filteredImmuneAccuracy_mean = round(sum(filteredImmuneAccuracy)/len(filteredImmuneAccuracy),4)

    performace_results_mean = {
                'Cell MAE': [round(sum(Absdiff_list_other)/len(Absdiff_list_other),4)],
                    'Cell MRE': [filteredCellAccuracy_mean],
                'Cell MRE max': [round(sum(AccuracyRelative_other)/len(AccuracyRelative_other),4)],
                'Cell RPD': [round(sum(AccuracyRelativePD_other)/len(AccuracyRelativePD_other),4)],
                'Cell Pearson r': [round(pearson_r_cell,4)],
                
                'Immune MAE': [round(sum(Absdiff_list_immune)/len(Absdiff_list_immune),4)],
                    'Immune MRE': [filteredImmuneAccuracy_mean],
                'Immune MRE max': [round(sum(AccuracyRelative_immune)/len(AccuracyRelative_immune),4)],
                'Immune RPD': [round(sum(AccuracyRelativePD_immune)/len(AccuracyRelativePD_immune),4)],
                'Immune Pearson r': [round(pearson_r_immune,4)],    

                    'Ratio MAE': [round(sum(Absdiff_list_ratio)/len(Absdiff_list_ratio),4)],
                    'Ratio MRE': [round(sum(accuracy_ratio)/len(accuracy_ratio),4)],
                'Ratio MRE max': [round(sum(AccuracyRelative_ratio)/len(AccuracyRelative_ratio),4)],
                'Ratio RPD': [round(sum(AccuracyRelativePD_ratio)/len(AccuracyRelativePD_ratio),4)],
                                    'Ratio pearson r':[round(pearson_r_ratio,4)]}
    
    perf_dt2 = pd.DataFrame(performace_results_mean)
    perf_dt2.to_csv(os.path.join(save_dir,'resultsDataMean.csv'), index=False)
    
    # Column names (corresponding to your returned values)
    columns = [
        "gmae_cell",
        "gmae_cellAccuracyRelative",
        "gmae_cellAccuracyRelativePD",
        "gmae_immune",
        "gmae_immuneAccuracyRelative",
        "gmae_immuneAccuracyRelativePD",
    ]

    # Convert to DataFrame
    df1 = pd.DataFrame(G1metrics, columns=columns)
    df2 = pd.DataFrame(G2metrics, columns=columns)
    df3 = pd.DataFrame(G3metrics, columns=columns)
    
    column_means1 = df1.mean().to_numpy()  # Or use .values for older pandas versions
    column_means2 = df2.mean().to_numpy()  # Or use .values for older pandas versions
    column_means3 = df3.mean().to_numpy()  # Or use .values for older pandas versions

    data = [column_means1, column_means2, column_means3]
    res = pd.DataFrame(data, columns=columns, index=["G(1)", "G(2)", "G(3)"])
    res.to_csv(os.path.join(save_dir,'resultsGridCount.csv'), index=True)
    
    arr_f1_immune /= len(sample_list)
    arr_prec_immune /= len(sample_list)
    arr_recall_immune /= len(sample_list)
    
    arr_f1_other /= len(sample_list)
    arr_prec_other /= len(sample_list)
    arr_recall_other /= len(sample_list)
    
    columns = [
        "prec_cell",
        "recall_cell",
        "f1_cell",
        "prec_immune",
        "recall_immune",
        "f1_immune"
    ]
        
    index=["sigma(5)", "sigma(20)", "sigma(5)_09", "sigma(20)_09"]
    
    sigma5prec_immune, sigma20prec_immune = np.mean(arr_prec_immune, axis=1)
    sigma5recall_immune, sigma20recall_immune = np.mean(arr_recall_immune, axis=1)
    sigma5f1_immune, sigma20f1_immune = np.mean(arr_f1_immune, axis=1)
    
    sigma5prec_other, sigma20prec_other = np.mean(arr_prec_other, axis=1)
    sigma5recall_other, sigma20recall_other = np.mean(arr_recall_other, axis=1)
    sigma5f1_other, sigma20f1_other = np.mean(arr_f1_other, axis=1)

    
    row1 = [sigma5prec_other, sigma5recall_other, sigma5f1_other , sigma5prec_immune, sigma5recall_immune, sigma5f1_immune]
    row2 = [sigma20prec_other, sigma20recall_other, sigma20f1_other, sigma20prec_immune, sigma20recall_immune, sigma20f1_immune]
    
    sigma5prec_immune, sigma20prec_immune = np.mean(arr_prec_immune[:,:-1], axis=1)
    sigma5recall_immune, sigma20recall_immune = np.mean(arr_recall_immune[:,:-1], axis=1)
    sigma5f1_immune, sigma20f1_immune = np.mean(arr_f1_immune[:,:-1], axis=1)

    sigma5prec_other, sigma20prec_other = np.mean(arr_prec_other[:,:-1], axis=1)
    sigma5recall_other, sigma20recall_other = np.mean(arr_recall_other[:,:-1], axis=1)
    sigma5f1_other, sigma20f1_other = np.mean(arr_f1_other[:,:-1], axis=1)

    row3 = [sigma5prec_other, sigma5recall_other, sigma5f1_other, sigma5prec_immune, sigma5recall_immune, sigma5f1_immune]
    row4 = [sigma20prec_other, sigma20recall_other, sigma20f1_other, sigma20prec_immune, sigma20recall_immune, sigma20f1_immune]    

    # Create DataFrame
    res = pd.DataFrame([row1, row2, row3, row4], columns=columns, index=index)

    # Save DataFrame as CSV
    res.to_csv(os.path.join(save_dir,'resultsMatching.csv'), index=True)
    
    delete_index = []
    for i in range(0,len(sample_list)):
        if gt_list_immune[i]<25 or pred_list_immune[i]<25:
            delete_index.append(i)
            
    for i in sorted(delete_index, reverse=True):
        del(sample_list[i])

        del(gt_list_immune[i])
        del(pred_list_immune[i])
        del(Absdiff_list_immune[i])
        del(accuracy_immune[i])
        del(AccuracyRelative_immune[i])
        del(AccuracyRelativePD_immune[i])
        
        del(gt_list_other[i])
        del(pred_list_other[i])
        del(Absdiff_list_other[i])
        del(accuracy_other[i])
        del(AccuracyRelative_other[i])
        del(AccuracyRelativePD_other[i])   
        
        del(gt_list_ratio[i])
        del(pred_list_ratio[i])
        del(Absdiff_list_ratio[i])
        del(accuracy_ratio[i])
        del(AccuracyRelative_ratio[i])
        del(AccuracyRelativePD_ratio[i]) 
        
    performace_results = {'sample name':sample_list,
                    'cell count Gold':gt_list_other,
                    'cell count Pred': pred_list_other,
                'cell abs diff': Absdiff_list_other,
                    'cell accuracy': accuracy_other,
                'cell accuracy RD': AccuracyRelative_other,
                'cell accuracy RD Perantage':AccuracyRelativePD_other,     
                'immune count Gold':gt_list_immune,
                'immune count Pred':pred_list_immune,
                'immune abs diff':Absdiff_list_immune,
                'immune accuracy':accuracy_immune,
                'immune accuracy RD':AccuracyRelative_immune,
                'immune accuracy RD Perantage':AccuracyRelativePD_immune,                            
                'ratio Gold':gt_list_ratio,
                'ratio Pred':pred_list_ratio,
                'ratio abs diff': Absdiff_list_ratio,
                'ratio accuracy': accuracy_ratio,
                'ratio accuracy RD': AccuracyRelative_ratio,
                'ratio accuracy RD Perantage':AccuracyRelativePD_ratio}

    perf_dt = pd.DataFrame(performace_results)
    perf_dt.to_csv(os.path.join(save_dir,'resultsDataFiltered.csv'), index=False)
            

    performace_results_mean_filtered = {
                'Cell Absolute Difference': [round(sum(Absdiff_list_other)/len(Absdiff_list_other),4)],
                    'Cell Accuracy': [round(sum(accuracy_other)/len(accuracy_other),4)],
                'Cell Accuracy RD': [round(sum(AccuracyRelative_other)/len(AccuracyRelative_other),4)],
                'Cell Accuracy RPD': [round(sum(AccuracyRelativePD_other)/len(AccuracyRelativePD_other),4)],
                'Immune Absolute Difference': [round(sum(Absdiff_list_immune)/len(Absdiff_list_immune),4)],
                    'Immune Accuracy': [round(sum(accuracy_immune)/len(accuracy_immune),4)],
                'Immune Accuracy RD': [round(sum(AccuracyRelative_immune)/len(AccuracyRelative_immune),4)],
                'Immune Accuracy RPD': [round(sum(AccuracyRelativePD_immune)/len(AccuracyRelativePD_immune),4)],     
                    'Ratio Absolute Difference': [round(sum(Absdiff_list_ratio)/len(Absdiff_list_ratio),4)],
                    'Ratio Accuracy': [round(sum(accuracy_ratio)/len(accuracy_ratio),4)],
                'Ratio Accuracy RD': [round(sum(AccuracyRelative_ratio)/len(AccuracyRelative_ratio),4)],
                'Ratio Accuracy RPD': [round(sum(AccuracyRelativePD_ratio)/len(AccuracyRelativePD_ratio),4)]}
    
    perf_dt2 = pd.DataFrame(performace_results_mean_filtered)
    perf_dt2.to_csv(os.path.join(save_dir,'resultsDataMeanFiltered.csv'), index=False)
    
    return performace_results_mean

def getPointsFromTsv(tsv_path):
    import glob
    files = glob.glob(tsv_path+'*.tsv')
    dataset={}
    for i, label in enumerate(files):
        labelName = label.split('.tsv')[0].split('.png-points')[0].split('/')[-1]
        labelName = labelName.split('-he')[0].split('-HE')[0].split('/')[-1]
        dataset[labelName] = label
    return dataset

def main():
    
    save_dir = 'denemeReg2'
    test_path = '/home/ocaki13/projects/serous/Datav2/processed/datasetv2_768/fold3/test/'
    tsv_path = '/home/ocaki13/projects/serous/Datav2/processed/datasetv2_768/tsv/'
    tsv_files = getPointsFromTsv(tsv_path)
    
    image_list = get_image_list(test_path)
    modelType = 'TransUnet' #Unet
    input_size = (768,768)
    use_cuda = True
    model_path = '/media/ocaki13/KINGSTON/results/SingleReg/seros_exp5_singleReg_fold3/seros_exp5_singleReg_fold3_seed11/models/best.pt'
    device = "cuda:0"
    dtype = torch.cuda.FloatTensor
    Num_Class = 2
    ch = 3
    
    if modelType == 'TransUnet':
        patch_size = 16
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = Num_Class
        config_vit.n_skip = 3
        config_vit.patches.size = (patch_size, patch_size)
        config_vit.patches.grid = (int(input_size[0]/patch_size), int(input_size[1]/patch_size))
        model = ViT_seg(config_vit, img_size=input_size[0], num_classes=Num_Class).cuda()
    elif modelType == 'Unet':
        model = UNet(3, Num_Class, 64,
                    use_cuda, False, 0.2)
        # model = UNet_attention(1, 4, 64,
        #                 use_cuda, False, 0.2)

    model.load_state_dict(torch.load(model_path))
    model.eval()


    #resDict = test_single_mc(model, device, input_size, ch, Num_Class, image_list, tsv_files, save_dir)
    resDict = test_single_reg(model, device, input_size, ch, Num_Class, image_list, tsv_files, save_dir)


    
if __name__ == '__main__':
    main()