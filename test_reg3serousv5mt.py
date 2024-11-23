import numpy as np
import cv2
import os
from tqdm import tqdm
import re
import torch
from Model import UNet, UNet_attention, UNet_multitask
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
from skimage.color import rgb2hed
from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import VisionTransformerMultitask as ViT_seg_MT
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






classDict = {
    1:'other',
    2:'immune'}

REFERENCE_PATH = '/home/ocaki13/UNet-Torch/color_normalizer.npy'
REF = np.load(REFERENCE_PATH)

NORMALIZER = staintools.StainNormalizer(method='macenko')
NORMALIZER.fit(REF)

def test_multiple_reg(model, device, input_size, ch, Num_Class, image_list, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    

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
            output_immune, output_other = model(img_input)
            #probs = F.softmax(outputs, dim=1)
            output_immune = torch.nn.functional.relu(output_immune)
            output_other = torch.nn.functional.relu(output_other)

            out_other = output_other.squeeze(0).squeeze(0)
            out_other = out_other.cpu().detach().numpy()
            
            output_immune = output_immune.squeeze(0).squeeze(0)
            output_immune = output_immune.cpu().detach().numpy()
            
            if imgHeight != input_size[0] or imgWidth != input_size[1]:
                pred_other = zoom(out_other, (imgHeight / input_size[0], imgWidth / input_size[1]), order=0)
                pred_immune = zoom(output_immune, (imgHeight / input_size[0], imgWidth / input_size[1]), order=0)

            else:
                pred_other = out_other
                pred_immune = output_immune
        pred_other = pred_other/200
        pred_immune = pred_immune/200

        # read gt mask
        mask_path = img_path[:img_path.rfind('.')] + '_label_other_reg.npy'
        mask_other = np.load(mask_path)

        mask_path = img_path[:img_path.rfind('.')] + '_label_immune_reg.npy'
        mask_immune = np.load(mask_path)
        
        ##### Count other cells #####
        cellCountGt_other = np.sum(mask_other)
        cellCountPred_other = np.sum(pred_other)
        ## calculate absolute difference
        cellCountAbsDiff_other = abs(cellCountGt_other-cellCountPred_other)
        ## calculate other accuracy
        otherAccuracy = round(cellCountAbsDiff_other/(cellCountGt_other+smoothening_factor),4)
        otherAccuracyRelative = round(cellCountAbsDiff_other/max(cellCountGt_other,cellCountPred_other),4)
        otherAccuracyRelativePD =round((2*cellCountAbsDiff_other)/(cellCountGt_other+cellCountPred_other),4)
        
        ## add to list
        gt_list_other.append(round(cellCountGt_other, 4))
        pred_list_other.append(round(cellCountPred_other, 4))
        Absdiff_list_other.append(round(cellCountAbsDiff_other,4))
        accuracy_other.append(round(otherAccuracy, 4))
        AccuracyRelative_other.append(round(otherAccuracyRelative, 4))
        AccuracyRelativePD_other.append(round(otherAccuracyRelativePD, 4))
        
        ##### Count immune cells #####
        cellCountGt_immune = np.sum(mask_immune)
        cellCountPred_immune = np.sum(pred_immune)
        ## calculate absolute difference
        cellCountAbsDiff_immune =abs(cellCountGt_immune-cellCountPred_immune)
        ## calculate other accuracy        
        immuneAccuracy = round(cellCountAbsDiff_immune/(cellCountGt_immune + smoothening_factor),4)
        immuneAccuracyRelative = round(cellCountAbsDiff_immune/(max(cellCountGt_immune,cellCountPred_immune)+smoothening_factor),4)
        immuneAccuracyRelativePD = round((2*cellCountAbsDiff_immune)/(cellCountGt_immune + cellCountPred_immune+smoothening_factor),4)
        
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
        # calculate abs difference
        abs_diff_ratio = abs(ratioGT-ratioPred)
        # calculate accuracy
        ratioAccuracy = round(abs_diff_ratio/(ratioGT+smoothening_factor),4)
        ratioAccuracyRelative = round(abs_diff_ratio/(max(ratioGT,ratioPred)+smoothening_factor),4)
        ratioAccuracyRelativePD = round((2*abs_diff_ratio)/(ratioGT+ratioPred+smoothening_factor),4)
                
        gt_list_ratio.append(ratioGT)
        pred_list_ratio.append(ratioPred)
        Absdiff_list_ratio.append(abs_diff_ratio)
        accuracy_ratio.append(ratioAccuracy)
        AccuracyRelative_ratio.append(ratioAccuracyRelative)
        AccuracyRelativePD_ratio.append(ratioAccuracyRelativePD)
        
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
            
    filteredCellAccuracy = [min(num, 5) for num in accuracy_other]
    filteredImmuneAccuracy = [min(num, 5) for num in accuracy_immune]
            
    filteredCellAccuracy_mean = round(sum(filteredCellAccuracy)/len(filteredCellAccuracy),4)
    filteredImmuneAccuracy_mean = round(sum(filteredImmuneAccuracy)/len(filteredImmuneAccuracy),4)

    performace_results_mean = {
                'Cell Absolute Difference': [round(sum(Absdiff_list_other)/len(Absdiff_list_other),4)],
                    'Cell Accuracy': [filteredCellAccuracy_mean],
                'Cell Accuracy RD': [round(sum(AccuracyRelative_other)/len(AccuracyRelative_other),4)],
                'Cell Accuracy RPD': [round(sum(AccuracyRelativePD_other)/len(AccuracyRelativePD_other),4)],
                
                'Immune Absolute Difference': [round(sum(Absdiff_list_immune)/len(Absdiff_list_immune),4)],
                    'Immune Accuracy': [filteredImmuneAccuracy_mean],
                'Immune Accuracy RD': [round(sum(AccuracyRelative_immune)/len(AccuracyRelative_immune),4)],
                'Immune Accuracy RPD': [round(sum(AccuracyRelativePD_immune)/len(AccuracyRelativePD_immune),4)],
                     
                    'Ratio Absolute Difference': [round(sum(Absdiff_list_ratio)/len(Absdiff_list_ratio),4)],
                    'Ratio Accuracy': [round(sum(accuracy_ratio)/len(accuracy_ratio),4)],
                'Ratio Accuracy RD': [round(sum(AccuracyRelative_ratio)/len(AccuracyRelative_ratio),4)],
                'Ratio Accuracy RPD': [round(sum(AccuracyRelativePD_ratio)/len(AccuracyRelativePD_ratio),4)]}
    
    perf_dt2 = pd.DataFrame(performace_results_mean)
    perf_dt2.to_csv(os.path.join(save_dir,'resultsDataMean.csv'), index=False)
    
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


def main():
    
    save_dir = 'rrr'
    test_path = '/home/ocaki13/projects/serous/Datav2/processed/datasetv2_768_reg/fold2/test/'
    image_list = get_image_list(test_path)
    modelType = 'TransUnet' #Unet
    input_size = (384,384)
    use_cuda = True
    model_path = 'traintestdeneme/traintestdeneme_seed25/models/epoch5.pt'
    device = "cuda:0"
    dtype = torch.cuda.FloatTensor
    Num_Class = 1
    ch = 3
    
    if modelType == 'TransUnet':
        patch_size = 16
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = Num_Class
        config_vit.n_skip = 3
        config_vit.patches.size = (patch_size, patch_size)
        config_vit.patches.grid = (int(input_size[0]/patch_size), int(input_size[1]/patch_size))
        model = ViT_seg_MT(config_vit, img_size=input_size[0], num_classes=Num_Class).cuda()
    elif modelType == 'Unet':
        model = UNet_multitask(ch, Num_Class, 64,
                    use_cuda, False, 0.2)
        # model = UNet_attention(1, 4, 64,
        #                 use_cuda, False, 0.2)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    resDict = test_multiple_reg(model, device, input_size, ch, Num_Class, image_list, save_dir)


    
if __name__ == '__main__':
    main()