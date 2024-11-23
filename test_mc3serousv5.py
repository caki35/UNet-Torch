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
        self.edList = []
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


    def _iouCalculate(self, maskGt, maskPred):
        y_true_area = np.sum((maskGt == 1))
        y_pred_area = np.sum((maskPred == 1))
        combined_area = y_true_area + y_pred_area
        intersection = np.sum((maskGt == 1) * (maskPred== 1))
        iou = (intersection + self.smoothening_factor) / \
        (combined_area - intersection + self.smoothening_factor)
        return iou
    
    def _iouCalculateBBox(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
    
    def _iouCalculateCircle(self,circle1, circle2):
        x0, y0, r0 = circle1
        x1, y1, r1 = circle2
        
        rr0 = r0 * r0;
        rr1 = r1 * r1;
        areaCircle1 = np.pi*rr0
        areaCircle2 = np.pi*rr1
        d = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        phi = (np.arccos((rr0 + (d * d) - rr1) / (2 * r0 * d))) * 2;
        theta = (np.arccos((rr1 + (d * d) - rr0) / (2 * r1 * d))) * 2;
        area1 = 0.5 * theta * rr1 - 0.5 * rr1 * np.sin(theta);
        area2 = 0.5 * phi * rr0 - 0.5 * rr0 * np.sin(phi);
        intersection = area1 + area2
        union = areaCircle1 + areaCircle2 - intersection
        iou = intersection/union
        return iou
    


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
    def __init__(self, save_dir, iou_thresh = 0.5):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.edList = []
        self.iou_thresh = iou_thresh
        self.smoothening_factor = 1e-6
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
        self.label_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in self.classDict:
            self.classRes[i] = {'tp':0,'fp':0,'fn':0,'tn':0}
        self.save_dir = save_dir
        

    def _findObjects(self, img):
        objectDict = {}
        offset= 0
        cellCounts = {}
        for cls in self.classDict:
            cellCounts[cls] = 0
        for cls in self.classDict:
            imgCls = np.zeros_like(img)
            imgCls[img==cls] = 1
            contours, _ = cv2.findContours(imgCls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cellCounts[cls] = len(contours)
            for i,contour in enumerate(contours):
                objectDict[i+offset] = {'contour':contour,'class':cls}
            # 0 1 2 3 4
            # i=4
            # offset = 5
            # start 5. index in next class
            offset = offset+len(contours)
        return objectDict, cellCounts[1], cellCounts[2]


    def _iouCalculate(self, maskGt, maskPred):
        y_true_area = np.sum((maskGt == 1))
        y_pred_area = np.sum((maskPred == 1))
        combined_area = y_true_area + y_pred_area
        intersection = np.sum((maskGt == 1) * (maskPred== 1))
        iou = (intersection + self.smoothening_factor) / \
        (combined_area - intersection + self.smoothening_factor)
        return iou
    
    def _iouCalculateBBox(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
    
    def _iouCalculateCircle(self,circle1, circle2):
        x0, y0, r0 = circle1
        x1, y1, r1 = circle2
        
        rr0 = r0 * r0;
        rr1 = r1 * r1;
        areaCircle1 = np.pi*rr0
        areaCircle2 = np.pi*rr1
        d = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        phi = (np.arccos((rr0 + (d * d) - rr1) / (2 * r0 * d))) * 2;
        theta = (np.arccos((rr1 + (d * d) - rr0) / (2 * r1 * d))) * 2;
        area1 = 0.5 * theta * rr1 - 0.5 * rr1 * np.sin(theta);
        area2 = 0.5 * phi * rr0 - 0.5 * rr0 * np.sin(phi);
        intersection = area1 + area2
        union = areaCircle1 + areaCircle2 - intersection
        iou = intersection/union
        return iou
    


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
        GtDict, cellCountGt, immuneCountGt = self._findObjects(gtImg)
        predDict, cellCountPred, immuneCountPred = self._findObjects(predImg)
        abs_diff_cell = abs(cellCountGt-cellCountPred)
        abs_diff_immune = abs(immuneCountGt-immuneCountPred)

        cellAccuracy = round(abs_diff_cell/(cellCountGt+self.smoothening_factor),4)
        cellAccuracyRelative = round(abs_diff_cell/max(cellCountGt,cellCountPred),4)
        cellAccuracyRelativePD =round((2*abs_diff_cell)/(cellCountGt+cellCountPred),4)

        immuneAccuracy = round(abs_diff_immune/(immuneCountGt+self.smoothening_factor),4)
        immuneAccuracyRelative = round(abs_diff_immune/(max(immuneCountGt,immuneCountPred)+self.smoothening_factor),4)
        immuneAccuracyRelativePD = round((2*abs_diff_immune)/(immuneCountGt+immuneCountPred+self.smoothening_factor),4)

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

        ratioGT = immuneCountGt/(cellCountGt+immuneCountGt)
        ratioPred = immuneCountPred/(cellCountPred+immuneCountPred)
        abs_diff_ratio = abs(ratioGT-ratioPred)
        ratioAccuracy = round(abs_diff_ratio/(ratioGT+self.smoothening_factor),4)
        ratioAccuracyRelative = round(abs_diff_ratio/(max(ratioGT,ratioPred)+self.smoothening_factor),4)
        ratioAccuracyRelativePD = round((2*abs_diff_ratio)/(ratioGT+ratioPred+self.smoothening_factor),4)
        
        self.ratio['GT'].append(ratioGT)
        self.ratio['Pred'].append(ratioPred)
        self.ratio['AbsDiff'].append(round(abs_diff_ratio,4))
        self.ratio['Accuracy'].append(ratioAccuracy)
        self.ratio['AccuracyRelative'].append(ratioAccuracyRelative)
        self.ratio['AccuracyRelativePD'].append(ratioAccuracyRelativePD)

        currentClassDict = {} #dictionary for keeping class based tp
        for i in self.classDict:
            currentClassDict[i] = {'tp':0,'TotalGT':0,'TotalPred':0}
        # iterate over ground truth and calculate gt
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
        
        #calculate number of prediction based on classes
        for pred in predDict:
            currentClassDict[predDict[pred]['class']]['TotalPred'] += 1
            
        #calculate class based tp,fp,tn for current image and add to sums
        for i in self.classDict:
            self.classRes[i]['tp'] += currentClassDict[i]['tp']
            self.classRes[i]['fp'] += (currentClassDict[i]['TotalPred'] - currentClassDict[i]['tp'])
            self.classRes[i]['fn'] += (currentClassDict[i]['TotalGT'] - currentClassDict[i]['tp'])


        self.recall.append(round(tp / len(GtDict), 4))
        self.precision.append(round(tp / (tp + fp), 4))
        self.f1.append(round(tp / (tp + (0.5 * (fp + fn))), 4))
        self._save_visuals(img_org, gtImg, predImg, (cellCountGt, immuneCountGt), (cellCountPred, immuneCountPred), self.save_dir)
    def save(self):
        performace_results = {'sample name':self.imageNames,'precision':self.precision,'recall':self.recall,
                              'f1':self.f1, 'cell count Gold':self.cellCounts['GT'],
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
        filteredCellAccuracy = [min(num, 5) for num in self.cellCounts['Accuracy']]
        filteredImmuneAccuracy = [min(num, 5) for num in self.immuneCounts['Accuracy']]

        # filteredCellAccuracy = list(filter(lambda num: num < 5, self.cellCounts['Accuracy']))
        # filteredImmuneAccuracy = list(filter(lambda num: num < 5, self.immuneCounts['Accuracy']))

        self.performace_results = {'precision':round(precision,4)*100,'recall':round(recall,4)*100,
                              'f1':round(f1score,4)*100, 'mean Precision':round(meanPrecision,4)*100,
                              'mean Recall':round(meanRecall,4)*100, 'mean f1':round(meanf1,4)*100,
                              'mean ED':round(meanED,2), 
                              'Cell Precesion': classPrecision[0]*100, 
                              'Cell Recall': classRecall[0]*100,
                              'Cell F1': classF1[0]*100,
                            'Cell Absolute Difference': round(sum( self.cellCounts['AbsDiff'])/len( self.cellCounts['AbsDiff']),4),
                              'Cell Accuracy': round(sum(filteredCellAccuracy)/len(filteredCellAccuracy),4),
                            'Cell Accuracy RD': round(sum( self.cellCounts['AccuracyRelative'])/len( self.cellCounts['AccuracyRelative']),4),
                            'Cell Accuracy RPD': round(sum( self.cellCounts['AccuracyRelativePD'])/len( self.cellCounts['AccuracyRelativePD']),4),

                              'Immune Precesion': classPrecision[1]*100,
                              'Immune Recall': classRecall[1]*100,
                              'Immune F1': classF1[1]*100,
                            'Immune Absolute Difference': round(sum( self.immuneCounts['AbsDiff'])/len( self.immuneCounts['AbsDiff']),4),
                              'Immune Accuracy': round(sum(filteredImmuneAccuracy)/len(filteredImmuneAccuracy),4),
                            'Immune Accuracy RD': round(sum( self.immuneCounts['AccuracyRelative'])/len( self.immuneCounts['AccuracyRelative']),4),
                            'Immune Accuracy RPD': round(sum( self.immuneCounts['AccuracyRelativePD'])/len( self.immuneCounts['AccuracyRelativePD']),4),
                            
                                'Ratio Absolute Difference': round(sum(self.ratio['AbsDiff'])/len( self.immuneCounts['AbsDiff']),4),
                              'Ratio Accuracy': round(sum(self.ratio['Accuracy'])/len(self.ratio['Accuracy']),4),
                            'Ratio Accuracy RD': round(sum(self.ratio['AccuracyRelative'])/len(self.ratio['AccuracyRelative']),4),
                            'Ratio Accuracy RPD': round(sum(self.ratio['AccuracyRelativePD'])/len(self.ratio['AccuracyRelativePD']),4)}
        
        perf_dt2 = pd.DataFrame(self.performace_results, index=[0])
        perf_dt2.to_csv(os.path.join(self.save_dir,'results.csv'), index=False)
        
        self.performace_results = {
                    'Cell Absolute Difference': round(sum( self.cellCounts['AbsDiff'])/len( self.cellCounts['AbsDiff']),4),
                        'Cell Accuracy': round(sum(filteredCellAccuracy)/len(filteredCellAccuracy),4),
                    'Cell Accuracy RD': round(sum( self.cellCounts['AccuracyRelative'])/len( self.cellCounts['AccuracyRelative']),4),
                    'Cell Accuracy RPD': round(sum( self.cellCounts['AccuracyRelativePD'])/len( self.cellCounts['AccuracyRelativePD']),4),
                    'Immune Absolute Difference': round(sum( self.immuneCounts['AbsDiff'])/len( self.immuneCounts['AbsDiff']),4),
                        'Immune Accuracy': round(sum(filteredImmuneAccuracy)/len(filteredImmuneAccuracy),4),
                    'Immune Accuracy RD': round(sum( self.immuneCounts['AccuracyRelative'])/len( self.immuneCounts['AccuracyRelative']),4),
                    'Immune Accuracy RPD': round(sum( self.immuneCounts['AccuracyRelativePD'])/len( self.immuneCounts['AccuracyRelativePD']),4),     
                        'Ratio Absolute Difference': round(sum(self.ratio['AbsDiff'])/len( self.immuneCounts['AbsDiff']),4),
                        'Ratio Accuracy': round(sum(self.ratio['Accuracy'])/len(self.ratio['Accuracy']),4),
                    'Ratio Accuracy RD': round(sum(self.ratio['AccuracyRelative'])/len(self.ratio['AccuracyRelative']),4),
                    'Ratio Accuracy RPD': round(sum(self.ratio['AccuracyRelativePD'])/len(self.ratio['AccuracyRelativePD']),4)}
        
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
                    'Cell Absolute Difference': round(sum(cellCountsFilteredAbsDiff)/len(cellCountsFilteredAbsDiff),4),
                        'Cell Accuracy': round(sum(cellCountsFilteredAccuracy)/len(cellCountsFilteredAccuracy),4),
                    'Cell Accuracy RD': round(sum(cellCountsFilteredRD)/len(cellCountsFilteredRD),4),
                    'Cell Accuracy RPD': round(sum(cellCountsFilteredRDP)/len(cellCountsFilteredRDP),4),

                    'Immune Absolute Difference': round(sum(immuneCountsFilteredAbsDiff)/len(immuneCountsFilteredAbsDiff),4),
                        'Immune Accuracy': round(sum(immuneCountsFilteredAccuracy)/len(immuneCountsFilteredAccuracy),4),
                    'Immune Accuracy RD': round(sum(immuneCountsFilteredRD)/len(immuneCountsFilteredRD),4),
                    'Immune Accuracy RPD': round(sum(immuneCountsFilteredRDP)/len(immuneCountsFilteredRDP),4),
                    
                        'Ratio Absolute Difference': round(sum(ratioFilteredAbsDiff)/len(ratioFilteredAbsDiff),4),
                        'Ratio Accuracy': round(sum(ratioFilteredAccuracy)/len(ratioFilteredAccuracy),4),
                    'Ratio Accuracy RD': round(sum(ratioFilteredRD)/len(ratioFilteredRD),4),
                    'Ratio Accuracy RPD': round(sum(ratioFilteredRDP)/len(ratioFilteredRDP),4)}
        
        perf_dt2 = pd.DataFrame(performace_results_filtered, index=[0])
        perf_dt2.to_csv(os.path.join(self.save_dir,'resultsFiltered.csv'), index=False)
        
    def getResults(self):
        return self.performace_results


def test_single_mc(model, device, input_size, ch, Num_Class, image_list, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if Num_Class == 3:
        res = Results2Class(save_dir)
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
        res.imageNames.append(image_name)

        res.compareImages(img_org, mask, pred)
    res.save()    
    resDict = res.getResults()
    return resDict
def main():
    
    save_dir = 'deneme2'
    test_path = '/home/ocaki13/projects/serous/Datav2/processed/datasetv2_768/fold3/test/'
    image_list = get_image_list(test_path)
    modelType = 'TransUnet' #Unet
    input_size = (768,768)
    use_cuda = True
    model_path = '/home/ocaki13/projects/serous/exps/datav2_exp3/seros_exp3_64TransUNet_fold3/seros_exp3_64TransUNet_fold3_seed35/epoch119.pt'
    device = "cuda:0"
    dtype = torch.cuda.FloatTensor
    Num_Class = 3
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


    resDict = test_single_mc(model, device, input_size, ch, Num_Class, image_list, save_dir)


    
if __name__ == '__main__':
    main()