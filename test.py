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
from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import torchvision.transforms.functional as TF
from scipy.stats import pearsonr
from CrowdMatching import CrowdMatchingTest, GMAE, countAccuracyMetric, CrowdMatchingTest2

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png', '.tif', '.PNG', '.tiff']

SCORE_MAP_FLAG = True
SAVE_PREDICTION = True

def NoiseFiltering(img, thresh=150):
    unique_labels = np.unique(img)
    for i in unique_labels:
        if i == 0:
            continue

        binary_img = np.zeros_like(img)
        binary_img[img == i] = 1
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
            if '_label' not in filename and 'gt_dot' not in filename:
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

def preprocessCrop(img_org, label, gt_dot, crop_size):        
    padding_h = img_org.shape[0] % crop_size
    padding_w = img_org.shape[1] % crop_size

    if padding_w != 0:
        padding_w = crop_size - padding_w
    if padding_h != 0:
        padding_h = crop_size - padding_h

    pad_left = padding_w//2
    pad_right = padding_w - pad_left
    pad_top = padding_h//2
    pad_bottom = padding_h - pad_top
    label = np.pad(label,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0)
    gt_dot = np.pad(gt_dot,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0)
    img_input = np.pad(img_org,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=255)  
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
    
    return img_input, label, gt_dot

def create_label_coordinates(dataPath, shape=(768,768)):
    img_label = np.zeros(shape, np.float64)
    data = pd.read_csv(dataPath, sep='\t')
    for index, row in data.iterrows():
        #x = int(np.rint(row['x']))-1
        #y = int(np.rint(row['y']))-1

        x = int(np.rint(row['x']/2))-1
        y = int(np.rint(row['y']/2))-1
        
        x = min(x, img_label.shape[1])
        x = max(x, 0)
        y = min(y, img_label.shape[0])
        y = max(y, 0)

        if row['class'] == 'Stroma':
            img_label[y, x] = 1
        elif row['class'] == 'normal':
            img_label[y, x] = 1
        elif row['class'] == 'Tumor':
            img_label[y, x] = 1
        elif row['class'] == 'Immune cells':
            img_label[y, x] = 1
        elif row['class'] in ['endothelium', 'Endothelial', 'Endothelium']:
            img_label[y, x] = 1
        else:
            img_label[y, x] = 1
    return img_label

class ResultsCC:
    def __init__(self, save_dir, save_img = False):
        self.save_dir = save_dir
        self.save_image = save_img
        self.imageNames = []
        self.recall = []
        self.precision = []
        self.f1 = []
        self.G1metrics = []
        self.G2metrics = []
        self.G3metrics = []
        self.GT = []
        self.Pred = []
        self.AbsDiff = []
        self.RelativeAccuracy = []
        self.sigma_list=[5, 20]
        self.sigma_thresh_list=list(np.arange(0.5,1, 0.05))
        self.arr_prec=np.zeros((len(self.sigma_list), len(self.sigma_thresh_list)))
        self.arr_recall=np.zeros((len(self.sigma_list), len(self.sigma_thresh_list)))
        self.arr_f1=np.zeros((len(self.sigma_list), len(self.sigma_thresh_list)))
        self.performace_results  = {}
        
    def _findObjects(self, img):

        # find counter
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #keep its total number
        cellCount = len(contours)
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
        objectLocations = (np.array(e_coord_x), np.array(e_coord_y))

        return objectLocations, cellCount    
    
    def _save_visuals(self, img_org, mask_img, prediction, cellCountsGT, cellCountsPred, save_dir):
        
        fig, axs = plt.subplots(1, 3)
        fig.set_figheight(12)
        fig.set_figwidth(30)
        if len(img_org.shape) == 3:
            axs[0].imshow(img_org)
            axs[0].title.set_text('image')
        else:
            axs[0].imshow(img_org, cmap='gray')
            axs[0].title.set_text('image')
        axs[1].imshow(mask_img)
        axs[1].title.set_text('label')
        fig.text(.51, .17, "cell: {}".format(cellCountsGT), ha='center',color="red")
        axs[2].imshow(prediction)
        axs[2].title.set_text('prediction')
        fig.text(.79, .17, "cell: {}".format(cellCountsPred), ha='center',color="red")
        fig.savefig(os.path.join(save_dir,self.imageNames[-1]))
        fig.clf()
        plt.close(fig)
    
    def compareImages(self, img_org, gtImg, predImg, gt_dot):
        # create dot image from ground-truth csv 
        #gt_dot = create_label_coordinates(tsv_path)
        # calculate gold counts
        cellCountGt = int(np.sum(gt_dot))

        predLocalization, cellCountPred = self._findObjects(predImg)
        abs_diff_cell, cellAccuracy, _, _ = countAccuracyMetric(cellCountGt, cellCountPred)

        # keep metrics for the image
        self.GT.append(cellCountGt)
        self.Pred.append(cellCountPred)
        self.AbsDiff.append(abs_diff_cell)
        self.RelativeAccuracy.append(cellAccuracy)
        
        e_dot = np.zeros_like(gt_dot)
        e_coord_x, e_coord_y = predLocalization
        for e_indx in range(len(e_coord_y)):
            e_dot[e_coord_y[e_indx], e_coord_x[e_indx]] = 1
            
        gmae_cell_res, _, _ = GMAE(1, gt_dot, e_dot)
        self.G1metrics.append(gmae_cell_res)
        
        gmae_cell_res, _, _ = GMAE(2, gt_dot, e_dot)
        self.G2metrics.append(gmae_cell_res)
        
        gmae_cell_res, _, _ = GMAE(3, gt_dot, e_dot)
        self.G3metrics.append(gmae_cell_res)
        
        arr_prec_current, arr_recall_current, arr_f1_current = CrowdMatchingTest(gt_dot, 
                                                                                 predLocalization, 
                                                                                 self.sigma_list,
                                                                                 self.sigma_thresh_list,
                                                                                 inputType='Coordinates')
        
        self.arr_prec += arr_prec_current
        self.arr_recall += arr_recall_current     
        self.arr_f1 += arr_f1_current
        prec_current, recall_current, f1_current = CrowdMatchingTest2(gt_dot, 
                                                                            predLocalization, 
                                                                            10)
        
        self.precision.append(prec_current)
        self.recall.append(recall_current)
        self.f1.append(f1_current)
        
        if self.save_image:
            self._save_visuals(img_org, gtImg, predImg, cellCountGt, cellCountPred, self.save_dir)
            
    def save(self):
        performace_results = {'sample name':self.imageNames,
                              'cell count Gold':self.GT,
                              'cell count Pred': self.Pred,
                            'cell abs diff': self.AbsDiff,
                              'cell accuracy': self.RelativeAccuracy,
                              'precision':self.precision,
                              'recall':self.recall,
                              'f1':self.f1}
        
        perf_dt = pd.DataFrame(performace_results)
        perf_dt.to_csv(os.path.join(self.save_dir,'resultsData.csv'), index=False)
        
        pearson_r_cell, _ = pearsonr(self.GT, self.Pred)        
        meanAbsDiff = sum(self.AbsDiff)/len(self.AbsDiff)
        meanRelativeAccuracy = sum(self.RelativeAccuracy)/len(self.RelativeAccuracy)

        g1Mean = sum(self.G1metrics)/len(self.G1metrics)
        g2Mean = sum(self.G2metrics)/len(self.G2metrics)
        g3Mean = sum(self.G3metrics)/len(self.G3metrics)
        
        precMean = sum(self.precision)/len(self.precision)
        recalMean = sum(self.recall)/len(self.recall)
        f1Mean = sum(self.f1)/len(self.f1)
        
        self.arr_f1 /= len(self.imageNames)
        self.arr_prec /= len(self.imageNames)
        self.arr_recall /= len(self.imageNames)
                
        columns = [
            "prec_cell",
            "recall_cell",
            "f1_cell"
        ]
        
        index=["sigma(5)", "sigma(20)", "sigma(5)_09", "sigma(20)_09"]
        
        sigma5prec, sigma20prec = np.mean(self.arr_prec, axis=1)
        sigma5recall, sigma20recall = np.mean(self.arr_recall, axis=1)
        sigma5f1, sigma20f1 = np.mean(self.arr_f1, axis=1)

        
        row1 = [sigma5prec, sigma5recall, sigma5f1]
        row2 = [sigma20prec, sigma20recall, sigma20f1]
        

        sigma5prec, sigma20prec = np.mean(self.arr_prec[:,:-1], axis=1)
        sigma5recall, sigma20recall = np.mean(self.arr_recall[:,:-1], axis=1)
        sigma5f1, sigma20f1 = np.mean(self.arr_f1[:,:-1], axis=1)

        row3 = [sigma5prec, sigma5recall, sigma5f1]
        row4 = [sigma20prec, sigma20recall, sigma20f1]    
    
        # Create DataFrame
        res = pd.DataFrame([row1, row2, row3, row4], columns=columns, index=index)

        # Save DataFrame as CSV
        res.to_csv(os.path.join(self.save_dir,'resultsMatching.csv'), index=True)

                # Column names (corresponding to your returned values)

        self.performace_results = {
            'precsion':round(precMean,4),
            'recall':round(recalMean,4),
            'f1':round(f1Mean,4),
            'MAE':round(meanAbsDiff,4), 
            'MRE':round(meanRelativeAccuracy,4), 
            'pearsonr':round(pearson_r_cell,4), 
            'GAME1':round(g1Mean,4), 
            'GAME2':round(g2Mean,4),
            'GAME3':round(g3Mean,4),
            'precsion sigma5':round(row1[0],4),
            'recall sigma5':round(row1[1],4),
            'f1 sigma5':round(row1[2],4),
            'precsion sigma5_9':round(row3[0],4),
            'recall sigma5_9':round(row3[0],4),
            'f1 sigma5_9':round(row3[0],4),
            'precsion sigma20':round(row2[0],4),
            'recall sigma20':round(row2[1],4),
            'f1 sigma20':round(row2[2],4),
        }
        # Convert to DataFrame
        res = pd.DataFrame([self.performace_results])
        res.to_csv(os.path.join(self.save_dir,'resultsCount.csv'), index=True)
    
        plt.scatter(self.GT, self.Pred, c ="black")
        plt.xlabel('golds')
        plt.ylabel('predictions')
        maxLimit = int(max(max(self.GT), max(self.Pred))) + 100
        plt.xlim(0,maxLimit)
        plt.ylim(0,maxLimit)
        plt.plot(range(maxLimit))
        plt.savefig(os.path.join(self.save_dir,'resultsData.png'))
        plt.cla()
        
    def getResults(self):
        return self.performace_results              
        

def test_single(model, device, input_size, ch, numClass, image_list, tsv_files, save_dir):
    res = ResultsCC(save_dir, True)
    
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
            out = model(img_input.to(device))
            out = torch.sigmoid(out)
            out = out.detach().data.cpu().numpy()
            out = out[0, 0]
            out[out >= 0.5] = 1
            out[out < 0.5] = 0
            if imgHeight != input_size[0] or imgWidth != input_size[1]:
                pred = zoom(out, (imgHeight / input_size[0], imgWidth / input_size[1]), order=0)
            else:
                pred = out
        pred = np.uint8(pred)

        # read gt mask
        mask_path = img_path[:img_path.rfind('.')] + '_label.png'
        mask = cv2.imread(mask_path, 0)
        
        img_name = img_path.split('/')[-1].split('.png')[0]
        tsv_path = tsv_files[img_name]
        
        res.imageNames.append(image_name)

        res.compareImages(img_org, mask, pred, tsv_path)
    res.save()    
    resDict = res.getResults()
    return resDict

def test_single_crop(model, device, input_size, ch, numClass, crop_size, image_list, save_dir):
    res = ResultsCC(save_dir, True)
    
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        img_org = cv2.imread(img_path)

        label_path =  img_path.replace('.png', '_label.png')
        gt_path =  img_path.replace('.png', '_gt_dot.png') 

        label = cv2.imread(label_path, 0)
        gt_dot = cv2.imread(gt_path, 0)
        
        img, label, gt_dot = preprocessCrop(img_org, label, gt_dot, crop_size)
                
        model.to(device=device)
        model.eval()
        pred = np.zeros_like(label)
        with torch.no_grad():
            for i in range(0, img.shape[2], crop_size):
                for j in range(0, img.shape[3], crop_size):
                    img_input = img[:,:,i:i+crop_size, j:j+crop_size]
                    out = model(img_input.to(device))
                    out = torch.sigmoid(out)
                    out = out.data.cpu().numpy()
                    out = out[0, 0]
                    out[out >= 0.5] = 1
                    out[out < 0.5] = 0
                    pred[i:i+crop_size, j:j+crop_size] = out
                    
        
        res.imageNames.append(image_name)
        res.compareImages(img_org, label, pred, gt_dot)
    res.save()    
    resDict = res.getResults()
    return resDict

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
    
    save_dir = 'resNew'
    test_path = '/home/ocaki13/projects/cellDatasets/processed/Dataset-BRCA-M2C/test/'
    image_list = get_image_list(test_path)

    tsv_path = '/home/ocaki13/projects/serous/Datav2/processed/datasetv2_768/tsv/'
    tsv_files = getPointsFromTsv(tsv_path)
        
    modelType = 'TransUnet' #Unet
    input_size = (256,256)
    use_cuda = True
    model_path = 'newDataset/newDataset_seed11/models/best.pt'
    device = "cuda:0"
    dtype = torch.cuda.FloatTensor
    Num_Class = 1
    ch = 3
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if modelType == 'TransUnet':
        patch_size = 16
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = Num_Class
        config_vit.n_skip = 3
        config_vit.patches.size = (patch_size, patch_size)
        config_vit.patches.grid = (int(input_size[0]/patch_size), int(input_size[1]/patch_size))
        model = ViT_seg(config_vit, img_size=input_size[0], num_classes=Num_Class).cuda()
    elif modelType == 'Unet':
        model = UNet(1, Num_Class, 64,
                    use_cuda, False, 0.2)
        # model = UNet_attention(1, 4, 64,
        #                 use_cuda, False, 0.2)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    #resultsDict = test_single(model, device, input_size, ch, Num_Class, image_list, tsv_files, save_dir)
    resultsDict = test_single_crop(model, device, input_size, ch, Num_Class, 256, image_list, save_dir)

if __name__ == "__main__":
    main()
    
    