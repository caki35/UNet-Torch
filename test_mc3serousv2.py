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


def pre_processGray(img):
    img = np.float32(img)
    img = (img - img.mean()) / img.std()
    # HW to CHW (for gray scale)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)

    # HWC to CHW, BGR to RGB (for three channel)
    # img = img.transpose((2, 0, 1))[::-1]
    img = torch.as_tensor(img)
    return img

def pre_process_rgb(img):
    img = np.float32(img)
    # img = (img - img.mean()) / img.std()
    # HW to CHW (for gray scale)
    img[:, :, 0] = (img[:, :, 0] - img[:, :, 0].mean()
                    ) / img[:, :, 0].std()
    img[:, :, 1] = (img[:, :, 1] - img[:, :, 1].mean()
                    ) / img[:, :, 1].std()
    img[:, :, 2] = (img[:, :, 2] - img[:, :, 2].mean()
                    ) / img[:, :, 2].std()

    # HWC to CHW, BGR to RGB (for three channel)
    img = img.transpose((2, 0, 1))[::-1]
    # add batch
    img = np.expand_dims(img, 0)
    img = torch.as_tensor(img.copy())

    return img


# use_cuda = True
# model_path = '/kuacc/users/ocaki13/hpc_run/workfolder/sereous_STMC_fold_bn_train234_val5_test1_exp1/epoch32.pt'
# model = UNet(1, 3, 64,
#              use_cuda, False, 0.2)

# # model = UNet_attention(1, 4, 64,
# #                 use_cuda, False, 0.2)


# model.load_state_dict(torch.load(model_path))
# model.eval()
# device = "cuda:0"
# dtype = torch.cuda.FloatTensor
# model.to(device=device)


patch_size = 16
img_size = (768,768)
use_cuda = True
model_path = '/kuacc/users/ocaki13/hpc_run/workfolder/epoch188.pt'
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


#test_path = '/kuacc/users/ocaki13/hpc_run/RT_resized/processed'
#test_path = '/kuacc/users/ocaki13/hpc_run/us_new_resized4/test'
test_path = '/kuacc/users/ocaki13/hpc_run/serousV1/fold0/test'
image_list = get_image_list(test_path)

save_dir = 'res'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


classDict = {
    1:'other',
    2:'immune'}

class Results:
    def __init__(self, classDict, save_dir, iou_thresh = 0.5):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.iou_list = []
        self.iou_thresh = iou_thresh
        self.smoothening_factor = 1e-6
        self.recall = []
        self.precision = []
        self.f1 = []
        self.classDict = classDict
        self.cellCounts = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.immuneCounts = {'GT':[],'Pred':[], 'Accuracy':[]}
        self.classRes = {}
        self.imageNames = []
        self.label_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0)]
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
                objectDict[i+offset] = {'contour':contour,'detected':0,'class':cls}
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
        fig.text(.51, .17, "cells: {}".format(cellCountsGtOthers), ha='center',color="red")
        fig.text(.51, .15, "immune {}".format(cellCountsGtImmune), ha='center',color="green")
        axs[2].imshow(self._create_rgb_mask(prediction))
        axs[2].title.set_text('prediction')
        fig.text(.79, .17, "cells: {}".format(cellCountsPredOther), ha='center',color="red")
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
        self.immuneCounts['GT'].append(immuneCountGt)
        
        self.cellCounts['Pred'].append(cellCountPred)
        self.immuneCounts['Pred'].append(immuneCountPred)

        self.cellCounts['Accuracy'].append(cellAccuracy)
        self.immuneCounts['Accuracy'].append(immuneAccuracy)

        
        for pred in predDict:
            # Segmentation mask iou
            #maskPred = np.zeros_like(predImg)
            # Draw the filled contour on the mask (inside will be 1)
            #cv2.drawContours(maskPred, [predDict[pred]['contour']], -1, 1, thickness=cv2.FILLED)
            
            #bbox iou
            xMinPred, yMinPred, wPred, hPred = cv2.boundingRect(predDict[pred]['contour']) 
            
            # #circle iou
            # (xPred,yPred),radiusPred = cv2.minEnclosingCircle(predDict[pred]['contour'])
            
            detectedFlag = False
            for gt in GtDict:
                # Segmentation mask iou
                #maskGt = np.zeros_like(gtImg)
                #cv2.drawContours(maskGt, [GtDict[gt]['contour']], -1, 1, thickness=cv2.FILLED)


                #bbox iou
                xMinGt, yMinGt, wGt, hGt = cv2.boundingRect(GtDict[gt]['contour']) 
                iou = self._iouCalculateBBox([xMinPred, yMinPred, xMinPred+wPred, yMinPred+hPred], 
                                                      [xMinGt, yMinGt, xMinGt+wGt, yMinGt+hGt])
                
                # # Circle iou
                # (xGt,yGt),radiusGT = cv2.minEnclosingCircle(GtDict[gt]['contour'])
                # iou = self._iouCalculateCircle([xPred, yPred, radiusPred], [xGt, yGt, radiusGT])
                
                if GtDict[gt]['detected']:
                    continue

                if iou> self.iou_thresh and predDict[pred]['class']==GtDict[gt]['class']:
                    detectedFlag = True
                    self.iou_list.append(iou)
                    GtDict[gt]['detected'] = 1

            if detectedFlag:
                self.tp += 1
                tp += 1
                self.classRes[predDict[pred]['class']]['tp'] +=1
            else:
                self.fp += 1
                self.classRes[predDict[pred]['class']]['fp'] +=1
                fp += 1

        for gt in GtDict:
            if GtDict[gt]['detected'] == 0:
                self.fn += 1
                fn += 1
                self.classRes[GtDict[gt]['class']]['fn'] +=1
                

        self.recall.append(round(tp / len(GtDict), 4))
        self.precision.append(round(tp / (tp + fp), 4))
        self.f1.append(round(tp / (tp + (0.5 * (fp + fn))), 4))
        self._save_visuals(img_org, gtImg, predImg, (cellCountGt, immuneCountGt), (cellCountPred, immuneCountPred), self.save_dir)
    def save(self):
        performace_results = {'sample name':self.imageNames,'precision':self.precision,'recall':self.recall,
                              'f1':self.f1, 'cell count Gold':self.cellCounts['GT'],
                              'cell count Pred': self.cellCounts['Pred'],
                              'cell count accuracy': self.cellCounts['Accuracy'],
                              'immune count Gold':self.immuneCounts['GT'],
                            'immune count Pred':self.immuneCounts['Pred'],
                            'immune accuracy':self.immuneCounts['Accuracy']}
        perf_dt = pd.DataFrame(performace_results)
        perf_dt.to_csv(os.path.join(self.save_dir,'resultsData.csv'), index=False)
        
        
        
        precision = res.tp/(res.tp+res.fp)
        recall = res.tp/(res.tp+res.fn)
        f1score = 2*precision*recall/(precision+recall)
        meanPrecision = sum(res.precision)/len(res.precision)
        meanRecall = sum(res.recall)/len(res.recall)
        meanf1 = sum(res.f1)/len(res.f1)
        meanIoU = sum(res.iou_list)/len(res.iou_list)
        
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
                              'mean IoU':round(meanIoU,4)*100, 
                              'Cell Precesion': classPrecision[0], 
                              'Cell Recall': classRecall[0],
                              'Cell F1': classF1[0],
                              'Immune Precesion': classPrecision[1],
                              'Immune Recall': classRecall[1],
                              'Immune F1': classF1[1]}
        perf_dt2 = pd.DataFrame(performace_results, index=[0])
        perf_dt2.to_csv(os.path.join(self.save_dir,'results.csv'), index=False)


res = Results(classDict, save_dir)
for img_path in tqdm(image_list):
    image_name = img_path.split('/')[-1]
    img_org = cv2.imread(img_path, 0)

    imgHeight, imgWidth = img_org.shape[0], img_org.shape[1]
    if imgHeight != img_size[0] or imgWidth != img_size[1]:
        img_resize = zoom(img_org, (img_size[0] / imgHeight, img_size[1] / imgWidth), order=3)  
        img_input = torch.from_numpy(img_resize).unsqueeze(0).unsqueeze(0).float().cuda()

    else:
        img_input = torch.from_numpy(img_org).unsqueeze(0).unsqueeze(0).float().cuda()

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
    print(pred.shape)
    print(np.unique(pred))

    print(mask.shape)

    res.imageNames.append(image_name)

    res.compareImages(img_org, mask, pred)
res.save()
