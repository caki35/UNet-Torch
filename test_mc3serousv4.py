import numpy as np
import cv2
import os
from tqdm import tqdm
import re
import torch
from Model import UNet, UNet_attention,UNet_BS
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


def preprocess(img_org):
    if len(img_org.shape)==2:
        imgHeight, imgWidth = img_org.shape
        if imgHeight != img_size[0] or imgWidth != img_size[1]:
            img_input = zoom(img_org, (img_size[0] / imgHeight, img_size[1] / imgWidth), order=3)  
        else:
            img_input = img_org
    else:
        imgHeight, imgWidth, c = img_org.shape
        if imgHeight != img_size[0] or imgWidth != img_size[1]:
            img_input = zoom(img_org, (img_size[0] / imgHeight, img_size[1] / imgWidth, 1), order=3)  
    #z normalizization
    mean3d = np.mean(img_input, axis=(0,1))
    std3d = np.std(img_input, axis=(0,1))
    img_input = (img_input-mean3d)/std3d
    if len(img_input.shape)==2:
        img_input = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float().cuda()
    else:
        img_input = img_input.transpose((2, 0, 1))[::-1]
        img_input = torch.from_numpy(img_input.astype(np.float32)).unsqueeze(0).float().cuda()
    
    # # Transform to tensor
    # img_input = TF.to_tensor(img_input)
    # # Normalize
    # img_input = TF.normalize(img_input,[0.5], [0.5]).unsqueeze(0).cuda()

    return img_input
# img_size = (768,768)
# use_cuda = True
# model_path = '/kuacc/users/ocaki13/hpc_run/workfolder/seroz_3cls_unet_exp2_rgb_fold1/epoch144.pt'
# model = UNet(3, 4, 64,
#              use_cuda, False, 0.2)

# # # model = UNet_attention(1, 4, 64,
# # #                 use_cuda, False, 0.2)


# model.load_state_dict(torch.load(model_path))
# model.eval()
# device = "cuda:0"
# dtype = torch.cuda.FloatTensor
# model.to(device=device)


patch_size = 16
img_size = (768,768)
use_cuda = True
model_path = '/kuacc/users/ocaki13/hpc_run/workfolder/res/seroz_2cls_TransUnet_exp3_macenko_fold2/epoch151.pt'
config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
config_vit.n_classes = 3
config_vit.n_skip = 3
config_vit.patches.size = (patch_size, patch_size)
config_vit.patches.grid = (int(img_size[0]/patch_size), int(img_size[1]/patch_size))
device = "cuda:0"
dtype = torch.cuda.FloatTensor
model = ViT_seg(config_vit, img_size=img_size[0], num_classes=4).cuda()
model.load_state_dict(torch.load(model_path))
model.eval()


test_path = '/kuacc/users/ocaki13/hpc_run/serousV1/fold2/test'
image_list = get_image_list(test_path)

save_dir = '/kuacc/users/ocaki13/hpc_run/workfolder/res/seroz_2cls_TransUnet_exp3_macenko_fold2/res'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)



classDict = {
    1:'other',
    2:'immune'}

REFERENCE_PATH = "/kuacc/users/ocaki13/hpc_run/workfolder/color_normalizer.npy"
REF = np.load(REFERENCE_PATH)

NORMALIZER = staintools.StainNormalizer(method='macenko')
NORMALIZER.fit(REF)

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
        self.classDict =  {1:'other',
                            2:'immune',
                            3:'tumor'}
        self.cellCounts = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.immuneCounts = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.ratio = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.classRes = {}
        self.imageNames = []
        self.label_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in self.classDict:
            self.classRes[i] = {'tp':0,'fp':0,'fn':0,'tn':0}
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
        GtDict, cellCountGt, immuneCountGt, tumorCountGT = self._findObjects(gtImg)
        predDict, cellCountPred, immuneCountPred, tumorCountPred = self._findObjects(predImg)

        tumorAccuracy = round(abs(tumorCountGT-tumorCountPred)/tumorCountGT,4)
        immuneAccuracy = round(abs(immuneCountGt-immuneCountPred)/immuneCountGt,4)

        self.cellCounts['GT'].append(tumorCountGT)
        self.immuneCounts['GT'].append(immuneCountGt)
        
        self.cellCounts['Pred'].append(tumorCountPred)
        self.immuneCounts['Pred'].append(immuneCountPred)

        self.cellCounts['Accuracy'].append(tumorAccuracy)
        self.immuneCounts['Accuracy'].append(immuneAccuracy)

        ratioGT = immuneCountGt/tumorCountGT
        ratioPred = immuneCountPred/tumorCountPred
        ratioAccuracy = round(abs(ratioGT-ratioPred)/ratioGT,4)
        self.ratio['GT'].append(ratioGT)
        self.ratio['Pred'].append(ratioPred)
        self.ratio['Accuracy'].append(ratioAccuracy)

        currentClassDict = {}
        for i in self.classDict:
            currentClassDict[i] = {'tp':0,'TotalGT':0,'TotalPred':0}
        for gt in GtDict:
            (xGt,yGt),radiusGT = cv2.minEnclosingCircle(GtDict[gt]['contour'])
            currentClassDict[GtDict[gt]['class']]['TotalGT'] += 1
            for pred in predDict:
                (xPred,yPred),radiusPred = cv2.minEnclosingCircle(predDict[pred]['contour'])
                currentED = distance.euclidean((xGt,yGt),(xPred, yPred))
                if currentED< 10 and predDict[pred]['class']==GtDict[gt]['class']:
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
        performace_results = {'sample name':self.imageNames,'precision':self.precision,'recall':self.recall,
                              'f1':self.f1, 'tumor count Gold':self.cellCounts['GT'],
                              'tumor count Pred': self.cellCounts['Pred'],
                              'tumor count accuracy': self.cellCounts['Accuracy'],
                              'immune count Gold':self.immuneCounts['GT'],
                            'immune count Pred':self.immuneCounts['Pred'],
                            'immune accuracy':self.immuneCounts['Accuracy'],
                            'ratio Gold':self.ratio['GT'],
                            'ratio Pred':self.ratio['Pred'],
                            'ratio accuracy':self.ratio['Accuracy']}
        perf_dt = pd.DataFrame(performace_results)
        perf_dt.to_csv(os.path.join(self.save_dir,'resultsData.csv'), index=False)
        
        
        
        precision = res.tp/(res.tp+res.fp)
        recall = res.tp/(res.tp+res.fn)
        f1score = 2*precision*recall/(precision+recall)
        meanPrecision = sum(res.precision)/len(res.precision)
        meanRecall = sum(res.recall)/len(res.recall)
        meanf1 = sum(res.f1)/len(res.f1)
        meanED = sum(res.edList)/len(res.edList)
        sns_hist = sns.histplot(res.edList)
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

        performace_results = {'precision':round(precision,4)*100,'recall':round(recall,4)*100,
                              'f1':round(f1score,4)*100, 'mean Precision':round(meanPrecision,4)*100,
                              'mean Recall':round(meanRecall,4)*100, 'mean f1':round(meanf1,4)*100,
                              'mean IoU':round(meanED,2), 
                              'Cell Precesion': classPrecision[0]*100, 
                              'Cell Recall': classRecall[0]*100,
                              'Cell F1': classF1[0]*100,
                              'Immune Precesion': classPrecision[1]*100,
                              'Immune Recall': classRecall[1]*100,
                              'Immune F1': classF1[1]*100,
                              'Immune Accuracy': sum(self.immuneCounts['Accuracy'])/len(self.immuneCounts['Accuracy']),
                                'Tumor Precesion': classPrecision[2]*100,
                              'Tumor Recall': classRecall[2]*100,
                              'Tumor F1': classF1[2]*100,
                              'Tumor Accuracy': sum(self.cellCounts['Accuracy'])/len(self.cellCounts['Accuracy']),
                              'Ratio Accuracy': sum(self.ratio['Accuracy'])/len(self.ratio['Accuracy'])}
        perf_dt2 = pd.DataFrame(performace_results, index=[0])
        perf_dt2.to_csv(os.path.join(self.save_dir,'results.csv'), index=False)


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
        self.cellCounts = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.immuneCounts = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.ratio = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.classRes = {}
        self.imageNames = []
        self.label_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in self.classDict:
            self.classRes[i] = {'tp':0,'fp':0,'fn':0,'tn':0}
        self.save_dir = save_dir
        

    def _findObjects(self, img):
        objectDict = {}
        clsList = np.unique(img)
        offset= 0
        cellCounts = {}
        for cls in self.classDict:
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
        
        cellAccuracy = round(abs(cellCountGt-cellCountPred)/cellCountGt,4)
        immuneAccuracy = round(abs(immuneCountGt-immuneCountPred)/immuneCountGt,4)
        
        self.cellCounts['GT'].append(cellCountGt)
        self.cellCounts['Pred'].append(cellCountPred)
        self.cellCounts['Accuracy'].append(cellAccuracy)
        
        self.immuneCounts['GT'].append(immuneCountGt)
        self.immuneCounts['Pred'].append(immuneCountPred)
        self.immuneCounts['Accuracy'].append(immuneAccuracy)

        ratioGT = immuneCountGt/cellCountGt
        ratioPred = immuneCountPred/cellCountPred
        ratioAccuracy = round(abs(ratioGT-ratioPred)/ratioGT,4)
        self.ratio['GT'].append(ratioGT)
        self.ratio['Pred'].append(ratioPred)
        self.ratio['Accuracy'].append(ratioAccuracy)

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
                if currentED< 10 and predDict[pred]['class']==GtDict[gt]['class']:
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
                              'f1':self.f1, 'tumor count Gold':self.cellCounts['GT'],
                              'tumor count Pred': self.cellCounts['Pred'],
                              'tumor count accuracy': self.cellCounts['Accuracy'],
                              'immune count Gold':self.immuneCounts['GT'],
                            'immune count Pred':self.immuneCounts['Pred'],
                            'immune accuracy':self.immuneCounts['Accuracy'],
                            'ratio Gold':self.ratio['GT'],
                            'ratio Pred':self.ratio['Pred'],
                            'ratio accuracy':self.ratio['Accuracy']}
        perf_dt = pd.DataFrame(performace_results)
        perf_dt.to_csv(os.path.join(self.save_dir,'resultsData.csv'), index=False)
        
        precision = res.tp/(res.tp+res.fp)
        recall = res.tp/(res.tp+res.fn)
        f1score = 2*precision*recall/(precision+recall)
        meanPrecision = sum(res.precision)/len(res.precision)
        meanRecall = sum(res.recall)/len(res.recall)
        meanf1 = sum(res.f1)/len(res.f1)
        meanED = sum(res.edList)/len(res.edList)
        sns_hist = sns.histplot(res.edList)
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

        performace_results = {'precision':round(precision,4)*100,'recall':round(recall,4)*100,
                              'f1':round(f1score,4)*100, 'mean Precision':round(meanPrecision,4)*100,
                              'mean Recall':round(meanRecall,4)*100, 'mean f1':round(meanf1,4)*100,
                              'mean ED':round(meanED,2), 
                              'Cell Precesion': classPrecision[0]*100, 
                              'Cell Recall': classRecall[0]*100,
                              'Cell F1': classF1[0]*100,
                              'Cell Accuracy': sum(self.cellCounts['Accuracy'])/len(self.cellCounts['Accuracy']),
                              'Immune Precesion': classPrecision[1]*100,
                              'Immune Recall': classRecall[1]*100,
                              'Immune F1': classF1[1]*100,
                              'Immune Accuracy': sum(self.immuneCounts['Accuracy'])/len(self.immuneCounts['Accuracy']),
                              'Ratio Accuracy': sum(self.ratio['Accuracy'])/len(self.ratio['Accuracy'])}
        perf_dt2 = pd.DataFrame(performace_results, index=[0])
        perf_dt2.to_csv(os.path.join(self.save_dir,'results.csv'), index=False)


#res = Results3Class(save_dir)
res = Results2Class(save_dir)

for img_path in tqdm(image_list):
    image_name = img_path.split('/')[-1]
    #img_org = cv2.imread(img_path)
    
    im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
    img_org = NORMALIZER.transform(im_rgb)
    # ihc_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
    # rihc_hed = rgb2hed(ihc_rgb)
    # img_org = rihc_hed[:,:,0]

    img_input = preprocess(img_org)
    imgHeight, imgWidth = img_org.shape[:2]
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_input)
        probs = F.softmax(outputs, dim=1)
        out = torch.argmax(probs, dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if imgHeight != img_size[0] or imgWidth != img_size[1]:
            pred = zoom(out, (imgHeight / img_size[0], imgWidth / img_size[1]), order=0)
        else:
            pred = out
    pred = np.uint8(pred)

    cv2.imwrite(os.path.join(save_dir, image_name+'_pred.png'),  pred)


    # read gt mask
    mask_path = img_path[:img_path.rfind('.')] + '_label_mc.png'
    mask = cv2.imread(mask_path, 0)
    res.imageNames.append(image_name)

    res.compareImages(img_org, mask, pred)
res.save()
