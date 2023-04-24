import numpy as np
import cv2
import os
from tqdm import tqdm
import re
import torch
from Model import UNet
import torch.nn.functional as F
import matplotlib.pyplot as plt

image_ext = ['.png', '.jpg', '.tif', '.tiff']


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
    img = np.expand_dims(img, 0)

    # HWC to CHW, BGR to RGB (for three channel)
    # img = img.transpose((2, 0, 1))[::-1]
    img = torch.as_tensor(img)
    return img


use_cuda = True
model_path = '/kuacc/users/ocaki13/hpc_run/workfolder/exp1_ultrasound/models/epoch83.pt'
model = UNet(1, 5, 64,
             use_cuda, True, 0.25)
model.load_state_dict(torch.load(model_path))
model.eval()
device = "cuda:0"
dtype = torch.cuda.FloatTensor
model.to(device=device)


val_path = '/kuacc/users/ocaki13/hpc_run/images/cizik'
image_list = get_image_list(val_path)

label_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


def create_rgb_mask(mask, label_colors):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask == 1] = label_colors[0]
    rgb_mask[mask == 2] = label_colors[1]
    rgb_mask[mask == 3] = label_colors[2]
    rgb_mask[mask == 4] = label_colors[3]

    return rgb_mask


save_dir = 'res'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


for img_path in tqdm(image_list):
    image_name = img_path.split('/')[-1]
    image_name = image_name[:image_name.rfind('')]
    img_org = cv2.resize(cv2.imread(
        img_path, 0), (1200, 800))

    img = pre_process(img_org)
    outputs = model(img.to(device))

    outputs = outputs.data.cpu().numpy()
    # (shape: (batch_size, img_h, img_w))
    pred_label_imgs = np.argmax(outputs, axis=1)
    pred_label_imgs = pred_label_imgs.astype(np.uint8)

    rgb_mask_pred = create_rgb_mask(pred_label_imgs[0], label_colors)
    # rgb_mask_pred = cv2.cvtColor(rgb_mask_pred, cv2.COLOR_BGR2RGB)

    # cv2.imwrite(os.path.join(save_dir, image_name+'.png'), rgb_mask_pred)

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(15)

    axs[0].imshow(img_org, cmap='gray')
    axs[0].title.set_text('image')
    axs[1].imshow(rgb_mask_pred)
    axs[1].title.set_text('label')

    fig.savefig(os.path.join(save_dir, image_name+'.png'))
    fig.clf()
    plt.close(fig)
