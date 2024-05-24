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


use_cuda = True
model_path = '/kuacc/users/ocaki13/hpc_run/workfolder/sereous_STMC_fold_bn_train234_val5_test1_exp1/epoch32.pt'
model = UNet(1, 3, 64,
             use_cuda, False, 0.2)

# model = UNet_attention(1, 4, 64,
#                 use_cuda, False, 0.2)


model.load_state_dict(torch.load(model_path))
model.eval()
device = "cuda:0"
dtype = torch.cuda.FloatTensor
model.to(device=device)


#test_path = '/kuacc/users/ocaki13/hpc_run/RT_resized/processed'
#test_path = '/kuacc/users/ocaki13/hpc_run/us_new_resized4/test'
test_path = '/kuacc/users/ocaki13/hpc_run/serozFolds/fold1/'
image_list = get_image_list(test_path)

save_dir = 'res'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

label_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0)]


def create_rgb_mask(mask, label_colors):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask == 1] = label_colors[0]
    rgb_mask[mask == 2] = label_colors[1]
    rgb_mask[mask == 3] = label_colors[2]

    return rgb_mask





class_names = {0: 'background', 1: 'red', 2: 'green'}

# red = 1;
# green = 2;
# yellow = 4;
# blue = 3;
class Results_mc:
    def __init__(self, save_dir, num_of_class, tolerance=0):
        self.save_dir = save_dir
        self.tolerance = tolerance
        self.num_of_class = num_of_class
        self.class_result_dict = {}
        for i in range(self.num_of_class):
            self.class_result_dict[i] = {'precision': [], 'recall': [
            ], 'f1': [], 'dice_score': [], 'iou_score': [], "hausdorff_distance":[]}

    def compare(self, y_true, y_pred):
        """
        calculate metrics threating each pixel as a sample
        """
        smoothening_factor = 1e-6
        for i in range(self.num_of_class):
            intersection = np.sum((y_pred == i) * (y_true == i))
            if (i not in y_true and i not in y_pred):
                self.class_result_dict[i]['iou_score'].append(1)
                self.class_result_dict[i]['dice_score'].append(1)
                self.class_result_dict[i]['precision'].append(1)
                self.class_result_dict[i]['recall'].append(1)
                self.class_result_dict[i]['f1'].append(1)
                self.class_result_dict[i]['hausdorff_distance'].append(0)
                continue

            y_true_area = np.sum((y_true == i))
            y_pred_area = np.sum((y_pred == i))
            combined_area = y_true_area + y_pred_area
            iou = (intersection + smoothening_factor) / \
                (combined_area - intersection + smoothening_factor)
            self.class_result_dict[i]['iou_score'].append(iou)

            dice_score = 2 * ((intersection + smoothening_factor) /
                              (combined_area + smoothening_factor))
            self.class_result_dict[i]['dice_score'].append(dice_score)

            # true positives
            tp_pp = np.sum((y_pred == i) & (y_true == i))
            # true negatives
            tn_pp = np.sum((y_pred == 0) & (y_true == 0))
            # false positives
            fp_pp = np.sum((y_pred == i) & (y_true != i))
            # false negatives
            fn_pp = np.sum((y_pred != i) & (y_true == i))

            precision = tp_pp / (tp_pp + fp_pp + smoothening_factor)
            recall = tp_pp / (tp_pp + fn_pp + smoothening_factor)
            f1_score = 2 * tp_pp / (2 * tp_pp + fp_pp + fn_pp)
            self.class_result_dict[i]['precision'].append(precision)
            self.class_result_dict[i]['recall'].append(recall)
            self.class_result_dict[i]['f1'].append(f1_score)

            y_binary_class_gt = np.zeros_like(y_true)
            y_binary_class_gt[y_true==i] = 1
            y_binary_class_pred = np.zeros_like(y_pred)
            y_binary_class_pred[y_pred==i] = 1
            if not np.all(y_binary_class_gt == 0) or np.all(y_binary_class_pred == 1):
                hausdorff = max(directed_hausdorff(y_binary_class_gt, y_binary_class_pred)[
                    0], directed_hausdorff(y_binary_class_pred, y_binary_class_gt)[0])
            self.class_result_dict[i]['hausdorff_distance'].append(hausdorff)

    def calculate_metrics(self):
        f = open(os.path.join(self.save_dir, 'result.txt'), 'w')

        f.write('Image-wise analysis:\n')

        for i in range(self.num_of_class):

            
            class_name = class_names[i]
            df = pd.DataFrame.from_dict(self.class_result_dict[i])
            df.to_csv(os.path.join(self.save_dir, '{}.csv'.format(class_name)), index=False)
            f.write('{} \n'.format(class_name))
            precision = round((sum(
                self.class_result_dict[i]['precision'])/len(self.class_result_dict[i]['precision']))*100, 2)
            recall = round((sum(
                self.class_result_dict[i]['recall'])/len(self.class_result_dict[i]['recall']))*100, 2)            
            f1_score = round((sum(
                self.class_result_dict[i]['f1'])/len(self.class_result_dict[i]['f1']))*100, 2)
            dice_score = round((sum(
                self.class_result_dict[i]['dice_score'])/len(self.class_result_dict[i]['dice_score']))*100, 2)
            iou_based_image = round((sum(
                self.class_result_dict[i]['iou_score'])/len(self.class_result_dict[i]['iou_score']))*100, 2)            

            hd_image = round(sum(
                self.class_result_dict[i]['hausdorff_distance'])/len(self.class_result_dict[i]['hausdorff_distance']), 2)
            
            f.write('precision: {}\n'.format(precision))
            f.write('recall: {}\n'.format(recall))
            f.write('f1: {}\n'.format(f1_score))
            f.write("Dice Score:"+str(dice_score)+'\n')
            f.write("IOU Score:" + str(iou_based_image)+'\n')
            f.write("Hausdorff Score:" + str(hd_image)+'\n')
            f.write("\n")
        f.close()

def count_cell(img):
    counter = label(img)
    counter_label_list = list(np.unique(counter))
    return len(counter_label_list) - 1


def countCellImage(img):
    binary_mask_cell = np.zeros_like(img)
    binary_mask_cell[img==1] = 1
    binary_mask_immune = np.zeros_like(img)
    binary_mask_immune[img==2] = 1
    countCell = count_cell(binary_mask_cell)
    countImmune = count_cell(binary_mask_immune)

    return [countCell, countImmune]

def save_visuals(img_org, mask_img, prediction, cellCountsGT, cellCountsPred, save_dir):
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
    axs[1].imshow(mask_img)
    axs[1].title.set_text('label')
    fig.text(.51, .17, "cells: {}".format(cellCountsGtOthers), ha='center',color="red")
    fig.text(.51, .15, "immune {}".format(cellCountsGtImmune), ha='center',color="green")
    axs[2].imshow(prediction)
    axs[2].title.set_text('prediction')
    fig.text(.79, .17, "cells: {}".format(cellCountsPredOther), ha='center',color="red")
    fig.text(.79, .15, "immune {}".format(cellCountsPredImmune), ha='center',color="green")
    fig.savefig(save_dir)
    fig.clf()
    plt.close(fig)

results = Results_mc(save_dir, 3)

def cellCountAccuracy(cellCountsPred, cellCountsGT):
    smoother = 1e-7 
    cellCountsGtOthers, cellCountsGtImmune = cellCountsGT
    cellCountsPredOther, cellCountsPredImmune = cellCountsPred


    accuracyCellOther = round(abs(cellCountsGtOthers - cellCountsPredOther)/(cellCountsGtOthers+smoother),4)
    accuracyCellImmune = round(abs(cellCountsGtImmune - cellCountsPredImmune)/(cellCountsGtImmune+smoother),4)
    accuracyCellTotal = round(abs((cellCountsGtOthers + cellCountsGtImmune) - (cellCountsPredImmune+cellCountsPredOther))/(cellCountsGtOthers + cellCountsGtImmune+smoother),4)

    return (accuracyCellOther, accuracyCellImmune, accuracyCellTotal)

performace_results = {
                "sample":[], "GT_other_cell": [], "GT_immune_cell": [], "Pred_other_cell": [],
                "Pred_immune_cell": [], "accuracy_cell": [], "accuracy_other": [], 
                'accuracy_immune': [], "gold_ratio":[],"pred_ratio":[],"ratio_accuracy":[]}

for img_path in tqdm(image_list):
    image_name = img_path.split('/')[-1]
    image_name = image_name[:image_name.rfind('.')]
    img_org = cv2.resize(cv2.imread(
        img_path, 0), (1536, 1536))

    # ihc_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
    # rihc_hed = rgb2hed(ihc_rgb)
    # img_org = rihc_hed[:,:,0]

    performace_results['sample'].append(image_name)
    if (len(img_org.shape)==2):
        img = pre_processGray(img_org)
    else:
        img = pre_process_rgb(img_org)
    outputs = model(img.to(device))
    probs = F.softmax(outputs, dim=1)

    probs = probs.data.cpu().numpy()
    # (shape: (batch_size, num_classes, img_h, img_w))
    outputs = outputs.data.cpu().numpy()
    # (shape: (batch_size, img_h, img_w))
    pred_label_imgs = np.argmax(outputs, axis=1)
    pred_label_imgs = pred_label_imgs.astype(np.uint8)
    pred_label_imgs = NoiseFiltering(pred_label_imgs, thresh=75)
    print(np.unique(pred_label_imgs[0]))
    # read dist mask
    mask_path = img_path[:img_path.rfind('.')] + '_label_mc.png'
    mask = cv2.resize(cv2.imread(
        mask_path, 0), (1536, 1536))
    
    cellCountsPred = countCellImage(pred_label_imgs)
    cellCountsGT = countCellImage(mask)

    performace_results["Pred_other_cell"].append(cellCountsPred[0])
    performace_results["Pred_immune_cell"].append(cellCountsPred[1])
    performace_results["GT_other_cell"].append(cellCountsGT[0])
    performace_results["GT_immune_cell"].append(cellCountsGT[1])

    accuracy = cellCountAccuracy(cellCountsPred, cellCountsGT)
    performace_results["accuracy_other"].append(accuracy[0])
    performace_results["accuracy_immune"].append(accuracy[1])
    performace_results["accuracy_cell"].append(accuracy[2])

    gold_ratio = cellCountsGT[1]/cellCountsGT[0]
    pred_ratio = cellCountsPred[1]/cellCountsPred[0]
    performace_results["gold_ratio"].append(gold_ratio)
    performace_results["pred_ratio"].append(pred_ratio)
    performace_results["ratio_accuracy"].append(abs(gold_ratio-pred_ratio)/(gold_ratio+1e-6))

    rgb_mask = cv2.cvtColor(create_rgb_mask(mask, label_colors), cv2.COLOR_RGB2BGR)
    rgb_mask_pred = cv2.cvtColor(create_rgb_mask(pred_label_imgs[0], label_colors), cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(save_dir, image_name+'_pred.png'),  rgb_mask_pred)

    score1 = probs[0, 1, :, :] * 255
    score2 = probs[0, 2, :, :] * 255
    # score3 = probs[0, 3, :, :] * 255

    score1_img = score1.astype(np.uint8)
    score2_img = score2.astype(np.uint8)
    # score3_img = score3.astype(np.uint8)

    results.compare(mask, pred_label_imgs[0])

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    axs[0].imshow(score1_img, cmap='gray')
    axs[0].title.set_text('class 1 (red)')
    axs[1].imshow(score2_img, cmap='gray')
    axs[1].title.set_text('class 2 (green)')
    # axs[2].imshow(score3_img, cmap='gray')
    # axs[2].title.set_text('class 3 (yellow)')
    # axs[1, 1].imshow(score4_img, cmap='gray')
    # axs[1, 1].title.set_text('class 4 (yellow)')
    fig.savefig(os.path.join(save_dir, image_name+'_probdist.png'))
    fig.clf()
    plt.close(fig)
    
    # seperater = np.zeros([img_org.shape[0], 15, 3], dtype=np.uint8)
    # seperater.fill(155)

    # save_img_dist = np.hstack(
    #     [img_org, seperater, mask_dist, seperater, pred_dist])
    # cv2.imwrite(os.path.join(results_save_dir_images,
    #             image_name+'_dist.png'), save_img_dist)

    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
    rgb_mask_pred = cv2.cvtColor(rgb_mask_pred, cv2.COLOR_BGR2RGB)
    save_visuals(img_org, rgb_mask, rgb_mask_pred, cellCountsGT, cellCountsPred, os.path.join(save_dir, image_name+'.png'))
results.calculate_metrics()

perf_dt = pd.DataFrame(performace_results)
perf_dt.to_csv(os.path.join(save_dir,'results.csv'), index=False)
accuracyImmune = perf_dt['accuracy_immune']
accuracyImmune = [45 if ele >
                    45 else ele for ele in accuracyImmune]

meanImmuneCountError = sum(
    accuracyImmune)/len(accuracyImmune)

accuracyOther = perf_dt['accuracy_other']
accuracyOther = [45 if ele >
                    45 else ele for ele in accuracyOther]
meanOtherCountError = sum(
    accuracyOther)/len(accuracyOther)

accuracyCounter = perf_dt['accuracy_cell']
accuracyCounter = [45 if ele >
                    45 else ele for ele in accuracyCounter]
meanCellCountError = sum(
    accuracyCounter)/len(accuracyCounter)

accuracyRatio = perf_dt['ratio_accuracy']
accuracyRatio = [45 if ele >
                    45 else ele for ele in accuracyRatio]
meanRatioError = sum(
    accuracyRatio)/len(accuracyRatio)
f = open(os.path.join(save_dir, 'result.txt'), 'w')
f.write('\n')
f.write('Accuracy Immune: ' + ' ' + str(meanImmuneCountError) + '\n')
f.write('Accuracy Other: ' + ' ' + str(meanOtherCountError) + '\n')
f.write('Accuracy Cell: ' + ' ' + str(meanCellCountError) + '\n')
f.write('Mean Ratio: ' + ' ' + str(meanRatioError) + '\n')
f.write('\n')