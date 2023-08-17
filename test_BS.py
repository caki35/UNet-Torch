from Model import UNet_BS, UNet_multitask
import torch
import os
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import re
from sklearn.metrics import confusion_matrix


image_ext = ['.png', '.jpg']
device = "cuda:0"

test_path = '/home/caki/desktop/projects/liver/liver_dataset_processed/test'
model_path = 'results/hd_exp1_bs_45/models/epoch91.pt'


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


def pre_process(img):
    img = np.float32(img)
    img = (img - img.mean()) / img.std()
    # HW to CHW (for gray scale)
    img = np.expand_dims(img, 0)
    # add batch
    img = np.expand_dims(img, 0)

    # HWC to CHW, BGR to RGB (for three channel)
    # img = img.transpose((2, 0, 1))[::-1]
    img = torch.as_tensor(img)
    return img


def post_process_binary(pred_bin):
    pred_bin = torch.sigmoid(pred_bin)
    pred_bin = pred_bin.data.cpu().numpy()
    pred_bin = pred_bin[0, 0]
    pred_bin[pred_bin >= 0.5] = 1
    pred_bin[pred_bin < 0.5] = 0
    return pred_bin


def performance_function(actual, prediction, performace_results, show=False):

    conf_matrix = confusion_matrix(actual.flatten(), prediction.flatten())

    # Binary Situation
    TP = conf_matrix[1][1]
    FP = conf_matrix[0][1]

    TN = conf_matrix[0][0]
    FN = conf_matrix[1][0]

    Sensitiviy = TP / (TP + FN)
    Specificitiy = TN / (TN + FP)

    Recall = TP / (TP + FN)
    if (TP + FP) == 0:
        Precision = 0.0000
        F_score = 0.0000
    else:
        Precision = TP / (TP + FP)
        F_score = 2 * ((Precision * Recall) / (Precision + Recall))

    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    Dice_coeff = (2 * TP) / ((2 * TP) + FP + FN)

    hausdorff = max(directed_hausdorff(actual, prediction)[
                    0], directed_hausdorff(prediction, actual)[0])

    IoU_score = TP / (TP + FP + FN)

    if show == True:
        print("F Score/Dice Score:", np.round(F_score, 4) * 100, "| IoU:", np.round(IoU_score, 4) * 100,
              "| Hausdorff:", np.round(hausdorff, 4))

    performace_results["F Score"].append(np.round(F_score, 4) * 100)
    performace_results["IoU"].append(np.round(IoU_score, 4) * 100)
    performace_results["Precision"].append(np.round(Precision, 4) * 100)
    performace_results["Recall"].append(np.round(Recall, 4) * 100)
    performace_results["Hausdorff"].append(np.round(hausdorff, 4))
    performace_results["DiceScore"].append(np.round(Dice_coeff, 4) * 100)


def main():
    image_list = get_image_list(test_path)

    model = UNet_BS([1, 32, 64, 128, 256, 512], "parameters", "dropout")
    # model = UNet_multitask(1, 1, 32, True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device=device)
    performace_results = {
        "Sample": [], "Precision": [], "Recall": [], "F Score": [], "Hausdorff": [], "IoU": [], 'DiceScore': []}
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        image_name = image_name[:image_name.rfind('.')]
        performace_results["Sample"].append(image_name)

        img_org = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = pre_process(img_org)

        mask_path = img_path[:img_path.rfind('.')] + '_label.png'
        mask_img = cv2.imread(mask_path, 0)*255

        _, gt_binary_mask = cv2.threshold(
            mask_img, 125, 1, cv2.THRESH_BINARY)

        pred_bin = model(img.to(device))
        pred_bin = torch.sigmoid(pred_bin)
        pred_bin = pred_bin.data.cpu().numpy()
        pred_bin = pred_bin[0, 0]
        pred_bin[pred_bin >= 0.5] = 1
        pred_bin[pred_bin < 0.5] = 0
        performance_function(gt_binary_mask, pred_bin,
                             performace_results, show=False)

    perf_dt = pd.DataFrame(performace_results)
    perf_dt.loc['General Performance'] = perf_dt.mean().round(2)
    perf_dt.to_csv('results2.csv', index=False)


if __name__ == '__main__':
    main()
