import torch
from torch.utils.data import Dataset
import os
import re
from torchvision import transforms
import numpy as np
import cv2
import random
import torchvision.transforms.functional as TF
from skimage.color import rgb2hed
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import staintools
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
np.bool = bool
import imgaug.augmenters as iaa
import torchio as tio
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd 

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png', '.tif', '.PNG', '.tiff']

def pad_image(samples, padding_w, padding_h, pad_value=0):
    pad_left = random.randint(0,padding_w)
    pad_right = padding_w - pad_left
    pad_top = random.randint(0,padding_h)
    pad_bottom = padding_h - pad_top
    outs = []
    for img in samples:
        if img.ndim == 2:  # Grayscale
            img = np.pad(img,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='constant',
                        constant_values=0)
            outs.append(img)
        elif img.ndim == 3:  # RGB or multi-channel
            img = np.pad(img,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='constant',
                        constant_values=255)
            outs.append(img)
           
    return outs


def PathologyAugmentationAug(sample, aug):
    image, label = sample['image'], sample['label']
    
    # Wrap the segmentation map
    segmap = SegmentationMapsOnImage(label, shape=image.shape)
    image_aug, segmap_aug = aug(images=[image], segmentation_maps=[segmap])
    
    return {'image': image_aug[0], 'label': segmap_aug[0].get_arr()}

def PathologyAugmentationAugHM2(sample, aug):
    image = sample['image']
    label1, label2 = sample['label']

    # Wrap the heatmap(s)
    heatmaps_obj = HeatmapsOnImage(np.stack((label1, label2),axis=-1), shape=image.shape)
    
    # Apply the augmentation
    image_aug, heatmaps_aug = aug(images=[image], heatmaps=[heatmaps_obj])
    
    # Extract the augmented image and heatmap(s)
    return {'image': image_aug[0], 'label': [heatmaps_aug[0].get_arr()[:,:,0], heatmaps_aug[0].get_arr()[:,:,1]]}
    
def PathologyAugmentationAugHM(sample, aug):
    image = sample['image']
    label = sample['label']

    # Wrap the heatmap(s)
    heatmaps_obj = HeatmapsOnImage(label, shape=image.shape)
    
    # Apply the augmentation
    image_aug, heatmaps_aug = aug(images=[image], heatmaps=[heatmaps_obj])
    
    # Extract the augmented image and heatmap(s)
    return {'image': image_aug[0], 'label': heatmaps_aug[0].get_arr()}
    
def RadiologyAugmentationTIO(sample, transforms_dict):
    image, label = sample['image'], sample['label']
    
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image,(0,-1))),  # Add channel and batch dim
        label=tio.LabelMap(tensor=np.expand_dims(label,(0,-1))) # Add channel and batch dim
    )  
    # Apply augmentations
    transform = tio.OneOf(transforms_dict)
    transformed_subject = transform(subject)
    
    transformed_image = transformed_subject["image"].data.numpy()[0,:,:,0]
    transformed_label = transformed_subject["label"].data.numpy()[0,:,:,0]
    sample = {'image': transformed_image, 'label': transformed_label}
    return sample



def random_rot_flip(sample_list):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    res_list = []
    for currentSample in sample_list:
        currentSample = np.rot90(currentSample, k)
        currentSample = np.flip(currentSample, axis=axis).copy()
        res_list.append(currentSample)
    return res_list


def random_rotate(sample_list):
    angle = np.random.randint(-20, 20)
    res_list = []
    for currentSample in sample_list:
        currentSample = ndimage.rotate(currentSample, angle, order=0, reshape=False)
        res_list.append(currentSample)
    return res_list

class Data_Reg_Binary(Dataset):
    def __init__(self, data_path, ch=1, anydepth=False, input_size=(512, 512), augmentation=False):
        super(Data_Reg_Binary, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.augmentation = augmentation
        self.anydepth = anydepth
        self.height = input_size[0]
        self.width = input_size[1]

    def transform(self, sample):
        labelBinary, labelReg = sample['label']
        image = sample['image'] 

        if self.augmentation:
            sample_list = [image, labelBinary, labelReg]
            if random.random() > 0.5:
                sample_list = random_rot_flip(sample_list)
            elif random.random() > 0.5:
                sample_list = random_rotate(sample_list)
            image, labelBinary, labelReg = sample_list
        
        if len(image.shape)==2:
            y, x = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y), order=3)  # why not 3?
                label1 = zoom(label1, (self.width / x, self.height / y), order=0)
                label2 = zoom(label2, (self.width / x, self.height / y), order=0)
        else:
            y, x, c = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y,1), order=3)  # why not 3?
                label1 = zoom(label1, (self.width / x, self.height / y), order=0)
                label2 = zoom(label2, (self.width / x, self.height / y), order=0)
            
        #z normalizization
        mean3d = np.mean(image, axis=(0,1))
        std3d = np.std(image, axis=(0,1))
        image = (image-mean3d)/std3d
        
        if len(image.shape)==2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:  
            # HWC to CHW, BGR to RGB (for three channel)
            image = image.transpose((2, 0, 1))[::-1]
            image = torch.from_numpy(image.astype(np.float32))

        #image = self.normalizeTorch(image.astype(np.float32))
        labelBinary = torch.from_numpy(labelBinary.astype(np.float32))
        labelReg = torch.from_numpy(labelReg.astype(np.float32)*200)
        label = [labelBinary, labelReg]
        sample = {'image': image, 'label': label}
        return sample

    def __getitem__(self, index):
        # read image
        imgPath = self.image_list[index]
        if self.channel==1:
            if self.anydepth: 
                image = cv2.imread(img_path, -1)
            else:
                image = cv2.imread(img_path, 0)
        elif self.channel==3:
            image = cv2.imread(img_path)
        elif self.channel==-1:
            #image = cv2.imread(img_path)
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            rihc_hed = rgb2hed(im_rgb)
            image = rihc_hed[:,:,0]
        elif self.channel==-2:
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            image = self.NORMALIZER.transform(im_rgb)

        # read target label mask
        label_path = imgPath[:imgPath.rfind('.')] + '_label.png'
        labelBin = cv2.imread(label_path, 0)
        
        label_path =  img_path[:img_path.rfind('.')] + '_label_reg.npy'
        labelReg = np.load(label_path)
        
        # print(np.sum(label))
        # print(label)
        sample = {'image': image, 'label': [labelBin, labelReg]}
        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                if '_label' in filename:
                    continue
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_paths.append(apath)
        return self.natural_sort(image_paths)

class Data_Reg(Dataset):
    def __init__(self, data_path, ch, anydepth, augmentation, input_size=(512, 512)):
        super(Data_Reg, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.anydepth = anydepth
        self.augmentation = augmentation
        self.height = input_size[0]
        self.width = input_size[1]
        if self.channel == -2:
            REFERENCE_PATH = '/home/ocaki13/UNet-Torch/color_normalizer.npy'
            REF = np.load(REFERENCE_PATH)

            self.NORMALIZER = staintools.StainNormalizer(method='macenko')
            self.NORMALIZER.fit(REF)
        if self.augmentation:
            # Define augmentation pipeline IMGAUG.
            self.aug = iaa.Sequential(iaa.SomeOf((0,2),[
                iaa.Affine(rotate=(-40, 40), mode="constant",cval=255),
                iaa.Affine(translate_px={"x": (-40, 40), "y": (-40, 40)}, mode="constant",cval=255),
                iaa.Fliplr(),
                iaa.Flipud(),
                iaa.OneOf([iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)]),
                iaa.OneOf([iaa.GaussianBlur(sigma=(0.1, 0.25)),
                iaa.MedianBlur(k=(3)),
                iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.8, 1.2))])
            ]))

            # Define augmentation pipeline IMGAUG.
            self.transforms_dict = {
                tio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=40): 0.1,
                tio.transforms.RandomElasticDeformation(num_control_points=7, locked_borders=2): 0.1,
                tio.transforms.RandomAnisotropy(axes=(1, 2), downsampling=(2, 4)): 0.1,
                tio.transforms.RandomBlur(): 0.1,
                tio.transforms.RandomGhosting(): 0.1,
                tio.transforms.RandomSpike(num_spikes = 1, intensity= (1, 2)): 0.1,
                tio.transforms.RandomBiasField(coefficients = 0.2, order= 3): 0.1,
                tio.RandomGamma(log_gamma=0.1): 0.1,
            }
            self._colorJitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.01)

        self.Counter = 0
        
    def transform(self, sample):
        label = sample['label']
        image = sample['image'] 

        if self.augmentation:
            # image_org = sample['image'] 
            # label_org =  sample['label']
            # label1_org = label_org[:,:,0]
            # label2_org = label_org[:,:,1]
            
            sample_list = [image, label]
            if random.random() > 0.5:
                sample_list = random_rot_flip(sample_list)
            elif random.random() > 0.5:
                sample_list = random_rotate(sample_list)
            image, label = sample_list
            
            
            # if random.random() > 0.25:
                
            #     # image_org = sample['image'] 
            #     # label_org =  sample['label']
            #     # label1_org = label_org[:,:,0]
            #     # label2_org = label_org[:,:,1]
            #     sample = PathologyAugmentationAugHM(sample, self.aug)
            #     image = sample['image'] 
            #     image = np.array(self._colorJitter(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
            #     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            #     label = sample['label']


            # ##### DEBUG ######            
            # # Create a vertical line as a separator
            # self.Counter += 1
            # label1 = label[:,:,0]
            # label2 = label[:,:,1]
            # separator_width = 10
            # separator = np.zeros((image.shape[0], separator_width, 3), dtype=np.uint8) 

            # # Concatenate the images with the separator
            # concatenated_image = np.hstack((image_org, separator, image))

            # # Save the concatenated image
            # cv2.imwrite(os.path.join('augload/','imgaug'+str(self.Counter)+'aug.png'),concatenated_image)

            # fig, axs = plt.subplots(2, 2)
            # fig.set_figheight(20)
            # fig.set_figwidth(20)

            # axs[0,0].imshow(label1_org)
            # axs[0,0].title.set_text('original other')
            # fig.text(0.25, 0.5, "count: {}".format(np.sum(label1_org)), ha='center', fontsize=16)

            # axs[0,1].imshow(label1)
            # axs[0,1].title.set_text('augmented other')
            # fig.text(0.75, 0.5, "count: {}".format(np.sum(label1)), ha='center', fontsize = 16)
            
            # axs[1,0].imshow(label2_org)
            # axs[1,0].title.set_text('original immune')
            # fig.text(0.25, 0.08, "count: {}".format(np.sum(label2_org)), ha='center', fontsize = 16)

            # axs[1,1].imshow(label2)
            # axs[1,1].title.set_text('augmented immune')
            # fig.text(0.75, 0.08, "count: {}".format(np.sum(label2)), ha='center', fontsize = 16)
            
            # fig.savefig(os.path.join('augload/','imgaug'+str(self.Counter)+'aug_label.png'))
            # fig.clf()
            # plt.close(fig)  
                

        
        if len(image.shape)==2:
            y, x = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y), order=3)  # why not 3?
                label = zoom(label, (self.width / x, self.height / y), order=0)
        else:
            y, x, c = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y,1), order=3)  # why not 3?
                label = zoom(label, (self.width / x, self.height / y), order=0)
            
        #z normalizization
        mean3d = np.mean(image, axis=(0,1))
        std3d = np.std(image, axis=(0,1))
        image = (image-mean3d)/std3d
        if len(image.shape)==2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:  
            # HWC to CHW, BGR to RGB (for three channel)
            image = image.transpose((2, 0, 1))[::-1]
            image = torch.from_numpy(image.astype(np.float32))

        #HWC to CHW
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label.astype(np.float32)*200)
        
        sample = {'image': image, 'label': label}
        return sample
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        if self.channel==1:
            if self.anydepth: 
                image = cv2.imread(img_path, -1)
            else:
                image = cv2.imread(img_path, 0)
        elif self.channel==3:
            image = cv2.imread(img_path)
        elif self.channel==-1:
            #image = cv2.imread(img_path)
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            rihc_hed = rgb2hed(im_rgb)
            image = rihc_hed[:,:,0]
        elif self.channel==-2:
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            image = self.NORMALIZER.transform(im_rgb)

        label_path =  img_path[:img_path.rfind('.')] + '_label_reg.npy'
        label = np.load(label_path).astype(np.float32)
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)

        return sample['image'], sample['label']

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for current_path in path:
            for maindir, subdir, file_name_list in os.walk(current_path):
                for filename in file_name_list:
                    if '_label' in filename:
                        continue
                    apath = os.path.join(maindir, filename)
                    ext = os.path.splitext(apath)[1]
                    if ext in image_ext:
                        image_paths.append(apath)
        return self.natural_sort(image_paths)

class Data_Reg_MT(Dataset):
    def __init__(self, data_path, ch, anydepth, augmentation, input_size=(512, 512)):
        super(Data_Reg_MT, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.anydepth = anydepth
        self.augmentation = augmentation
        self.height = input_size[0]
        self.width = input_size[1]
        # self.transform = transforms.Compose(
        #                            [RandomGenerator(output_size=[input_size[0], input_size[1]])])
        # self.normalizeTorch = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
        # ])
        if self.channel == -2:
            REFERENCE_PATH = '/home/ocaki13/UNet-Torch/color_normalizer.npy'
            REF = np.load(REFERENCE_PATH)

            self.NORMALIZER = staintools.StainNormalizer(method='macenko')
            self.NORMALIZER.fit(REF)
            
        if self.augmentation:
            # Define augmentation pipeline IMGAUG.
            self.aug = iaa.Sequential(iaa.SomeOf((0,2),[
                iaa.Affine(rotate=(-40, 40), mode="constant",cval=255),
                iaa.Affine(translate_px={"x": (-40, 40), "y": (-40, 40)}, mode="constant",cval=255),
                iaa.Fliplr(),
                iaa.Flipud(),
                iaa.OneOf([iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)]),
                iaa.OneOf([iaa.GaussianBlur(sigma=(0.1, 0.25)),
                iaa.MedianBlur(k=(3)),
                iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.8, 1.2))])
            ]))

            # Define augmentation pipeline IMGAUG.
            self.transforms_dict = {
                tio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=40): 0.1,
                tio.transforms.RandomElasticDeformation(num_control_points=7, locked_borders=2): 0.1,
                tio.transforms.RandomAnisotropy(axes=(1, 2), downsampling=(2, 4)): 0.1,
                tio.transforms.RandomBlur(): 0.1,
                tio.transforms.RandomGhosting(): 0.1,
                tio.transforms.RandomSpike(num_spikes = 1, intensity= (1, 2)): 0.1,
                tio.transforms.RandomBiasField(coefficients = 0.2, order= 3): 0.1,
                tio.RandomGamma(log_gamma=0.1): 0.1,
            }
            self._colorJitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.01)

        self.Counter = 0
    def transform(self, sample):
        label1, label2 = sample['label']
        image = sample['image'] 

        if self.augmentation:
            if random.random() > 0.25:
                
                image_org = sample['image'] 
                label1_org, label2_org =  sample['label']
                sample = PathologyAugmentationAugHM2(sample, self.aug)
                image = sample['image'] 
                image = np.array(self._colorJitter(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                label1, label2 = sample['label']

                # ##### DEBUG ######            
                # # Create a vertical line as a separator
                # self.Counter += 1
                # separator_width = 10
                # separator = np.zeros((image.shape[0], separator_width, 3), dtype=np.uint8) 

                # # Concatenate the images with the separator
                # concatenated_image = np.hstack((image_org, separator, image))

                # # Save the concatenated image
                # cv2.imwrite(os.path.join('augload/','imgaug'+str(self.Counter)+'aug.png'),concatenated_image)

                # fig, axs = plt.subplots(2, 2)
                # fig.set_figheight(20)
                # fig.set_figwidth(20)

                # axs[0,0].imshow(label1_org)
                # axs[0,0].title.set_text('original other')
                # fig.text(0.25, 0.5, "count: {}".format(np.sum(label1_org)), ha='center', fontsize=16)

                # axs[0,1].imshow(label1)
                # axs[0,1].title.set_text('augmented other')
                # fig.text(0.75, 0.5, "count: {}".format(np.sum(label1)), ha='center', fontsize = 16)
                
                # axs[1,0].imshow(label2_org)
                # axs[1,0].title.set_text('original immune')
                # fig.text(0.25, 0.08, "count: {}".format(np.sum(label2_org)), ha='center', fontsize = 16)

                # axs[1,1].imshow(label2)
                # axs[1,1].title.set_text('augmented immune')
                # fig.text(0.75, 0.08, "count: {}".format(np.sum(label2)), ha='center', fontsize = 16)
                
                # fig.savefig(os.path.join('augload/','imgaug'+str(self.Counter)+'aug_label.png'))
                # fig.clf()
                # plt.close(fig)  
                
            # sample_list = [image, label1, label2]
            # if random.random() > 0.5:
            #     sample_list = random_rot_flip(sample_list)
            # elif random.random() > 0.5:
            #     sample_list = random_rotate(sample_list)
            # image, label1, label2 = sample_list
        
        if len(image.shape)==2:
            y, x = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y), order=3)  # why not 3?
                label1 = zoom(label1, (self.width / x, self.height / y), order=0)
                label2 = zoom(label2, (self.width / x, self.height / y), order=0)
        else:
            y, x, c = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y,1), order=3)  # why not 3?
                label1 = zoom(label1, (self.width / x, self.height / y), order=0)
                label2 = zoom(label2, (self.width / x, self.height / y), order=0)
            
        #z normalizization
        mean3d = np.mean(image, axis=(0,1))
        std3d = np.std(image, axis=(0,1))
        image = (image-mean3d)/std3d
        
        if len(image.shape)==2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:  
            # HWC to CHW, BGR to RGB (for three channel)
            image = image.transpose((2, 0, 1))[::-1]
            image = torch.from_numpy(image.astype(np.float32))

        #image = self.normalizeTorch(image.astype(np.float32))
        label1 = torch.from_numpy(label1.astype(np.float32)*200)
        label2 = torch.from_numpy(label2.astype(np.float32)*200)
        label = [label1, label2]
        sample = {'image': image, 'label': label}
        return sample
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        if self.channel==1:
            if self.anydepth: 
                image = cv2.imread(img_path, -1)
            else:
                image = cv2.imread(img_path, 0)
        elif self.channel==3:
            image = cv2.imread(img_path)
        elif self.channel==-1:
            #image = cv2.imread(img_path)
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            rihc_hed = rgb2hed(im_rgb)
            image = rihc_hed[:,:,0]
        elif self.channel==-2:
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            image = self.NORMALIZER.transform(im_rgb)

        label_path_immune =  img_path[:img_path.rfind('.')] + '_label_immune_reg.npy'
        label_path_other =  img_path[:img_path.rfind('.')] + '_label_other_reg.npy'
        label_immune = np.load(label_path_immune).astype(np.float32)
        label_other = np.load(label_path_other).astype(np.float32)
       
        # print(np.sum(label))
        # print(label)
        sample = {'image': image, 'label': [label_immune, label_other]}
        sample = self.transform(sample)
        # print(torch.sum(sample['label']))
        # print(sample['label'])
        
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for current_path in path:
            for maindir, subdir, file_name_list in os.walk(current_path):
                for filename in file_name_list:
                    if '_label' in filename:
                        continue
                    apath = os.path.join(maindir, filename)
                    ext = os.path.splitext(apath)[1]
                    if ext in image_ext:
                        image_paths.append(apath)
        return self.natural_sort(image_paths)

class Data_Binary(Dataset):
    def __init__(self, data_path, ch, anydepth, augmentation, input_size=(512, 512)):
        super(Data_Binary, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.anydepth = anydepth
        self.augmentation = augmentation
        self.Counter = 0

        self.height = input_size[0]
        self.width = input_size[1]

        if self.channel == -2:
            REFERENCE_PATH = '/home/ocaki13/UNet-Torch/color_normalizer.npy'
            REF = np.load(REFERENCE_PATH)

            self.NORMALIZER = staintools.StainNormalizer(method='macenko')
            self.NORMALIZER.fit(REF)
        
    def transform(self, sample):
        img_unnormalized, label, gt_dot_map = sample['image'], sample['label'], sample['gt_dot']
        if self.augmentation:
            sample_list = [img_unnormalized, label, gt_dot_map]
            if random.random() > 0.5:
                sample_list = random_rot_flip(sample_list)
            elif random.random() > 0.5:
                sample_list = random_rotate(sample_list)
            img_unnormalized, label, gt_dot_map = sample_list

        if len(img_unnormalized.shape)==2:
            y, x = img_unnormalized.shape
            if x != self.width or y != self.height:
                img_unnormalized = zoom(img_unnormalized, (self.width / x, self.height / y), order=3)  # why not 3?
                label = zoom(label, (self.width / x, self.height / y), order=0)
                gt_dot_map = zoom(gt_dot_map, (self.width / x, self.height / y), order=0)

        else:
            y, x, c = img_unnormalized.shape
            if x != self.width or y != self.height:
                img_unnormalized = zoom(img_unnormalized, (self.width / x, self.height / y,1), order=3)  # why not 3?
                label = zoom(label, (self.width / x, self.height / y), order=0)
                gt_dot_map = zoom(gt_dot_map, (self.width / x, self.height / y), order=0)

            
        #z normalizization
        mean3d = np.mean(img_unnormalized, axis=(0,1))
        std3d = np.std(img_unnormalized, axis=(0,1))
        image = (img_unnormalized-mean3d)/std3d
        if len(image.shape)==2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            img_unnormalized = torch.from_numpy(img_unnormalized.astype(np.float32)).unsqueeze(0)
        else:  
            # HWC to CHW, BGR to RGB (for three channel)
            image = image.transpose((2, 0, 1))[::-1]
            image = torch.from_numpy(image.astype(np.float32))
            
            # HWC to CHW, BGR to RGB (for three channel)
            img_unnormalized = img_unnormalized.transpose((2, 0, 1))[::-1]
            img_unnormalized = torch.from_numpy(img_unnormalized.astype(np.float32))

        #image = self.normalizeTorch(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'gt_dot':gt_dot_map, 'img_org':img_unnormalized}
        return sample
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        if self.channel==1:
            if self.anydepth: 
                img_org = cv2.imread(img_path, -1)
            else:
                img_org = cv2.imread(img_path, 0)
        elif self.channel==3:
            img_org = cv2.imread(img_path)
        elif self.channel==-1:
            #image = cv2.imread(img_path)
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            rihc_hed = rgb2hed(im_rgb)
            img_org = rihc_hed[:,:,0]
        elif self.channel==-2:
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            img_org = self.NORMALIZER.transform(im_rgb)

        label_path =  img_path.replace('.png', '_label_mc.png')
        gt_path =  img_path.replace('.png', '_gt_dot.png') 

        label = cv2.imread(label_path, 0)
        gt_dot = cv2.imread(gt_path, 0)
        
        sample = {'image': img_org, 'label': label, 'gt_dot':gt_dot}
        sample = self.transform(sample)

        return sample['image'], sample['label']
    
    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for current_path in path:
            for maindir, subdir, file_name_list in os.walk(current_path):
                for filename in file_name_list:
                    if '_label' in filename or '_gt_dot' in filename:
                        continue
                    apath = os.path.join(maindir, filename)
                    ext = os.path.splitext(apath)[1]
                    if ext in image_ext:
                        image_paths.append(apath)
        return self.natural_sort(image_paths)

class DataPointReg(Dataset):
    def __init__(self, data_path, point_files, ch, anydepth, augmentation, crop_size, num_knn, train):
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.anydepth = anydepth
        self.augmentation = augmentation
        self.pointFiles = point_files
        self.crop_size = crop_size
        self.num_knn = num_knn
        self.train = train
        if self.channel == -2:
            REFERENCE_PATH = '/home/ocaki13/UNet-Torch/color_normalizer.npy'
            REF = np.load(REFERENCE_PATH)

            self.NORMALIZER = staintools.StainNormalizer(method='macenko')
            self.NORMALIZER.fit(REF)
            
    def crop(self,img,label):
        
        crop_size_r = random.randint(0, img.shape[1] - self.crop_size)
        crop_size_c = random.randint(0, img.shape[2] - self.crop_size)
        '''crop image'''
        img_croped = img[:, crop_size_r: crop_size_r + self.crop_size, crop_size_c:crop_size_c + self.crop_size]
        '''crop kpoint'''
        label_croped = label[crop_size_r: crop_size_r + self.crop_size, crop_size_c:crop_size_c + self.crop_size]
        return img_croped, label_croped

        
        
    def transform(self, image, g_dot):
        if self.augmentation:
            sample_list = [image, g_dot]
            if random.random() > 0.5:
                sample_list = random_rot_flip(sample_list)
            elif random.random() > 0.5:
                sample_list = random_rotate(sample_list)
            image, g_dot = sample_list
            
        #z normalizization
        mean3d = np.mean(image, axis=(0,1))
        std3d = np.std(image, axis=(0,1))
        image = (image-mean3d)/std3d
        if len(image.shape)==2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:  
            # HWC to CHW, BGR to RGB (for three channel)
            image = image.transpose((2, 0, 1))[::-1]
            image = torch.from_numpy(image.astype(np.float32))

        return image, g_dot 

    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        if self.channel==1:
            if self.anydepth: 
                image = cv2.imread(img_path, -1)
            else:
                image = cv2.imread(img_path, 0)
        elif self.channel==3:
            image = cv2.imread(img_path)
        elif self.channel==-1:
            #image = cv2.imread(img_path)
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            rihc_hed = rgb2hed(im_rgb)
            image = rihc_hed[:,:,0]
        elif self.channel==-2:
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            image = self.NORMALIZER.transform(im_rgb)

        img_name = img_path.split('/')[-1].split('.png')[0]
        tsv_path = self.pointFiles[img_name]
        
        gt_dot = self.create_label_coordinates(tsv_path)
        # augmentation
        # image normalization, chw and tensor transform  
        image, gt_dot = self.transform(image, gt_dot)

        #crop img and dot annotation
        if self.train:
            img_patch, gt_dot_patch = self.crop(image, gt_dot)   
            '''num_points and points'''
            num_points = int(np.sum(gt_dot_patch))
            '''points'''
            gt_points = np.nonzero(torch.from_numpy(gt_dot_patch))
            distances = self.caculate_knn_distance(gt_points, num_points)
            points = torch.cat([gt_points, distances], dim=1)
            target = {}
            target['labels'] = torch.ones([1, num_points]).squeeze(0).type(torch.LongTensor)
            target['points_macher'] = torch.true_divide(points, self.crop_size).type(torch.FloatTensor)
            target['points'] = torch.true_divide(points[:, 0:3], self.crop_size).type(torch.FloatTensor)
            return [img_patch], [target]
        else:
            gt_dot = torch.from_numpy(gt_dot).unsqueeze(0).cuda()

            width, height = image.shape[2], image.shape[1]
            num_w = int(width / self.crop_size)
            num_h = int(height / self.crop_size)
            '''image to patch'''
            img_return = image.view(3, num_h, self.crop_size, width).view(3, num_h, self.crop_size, num_w,
                                                                                self.crop_size)
            img_return = img_return.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, self.crop_size,
                                                                             self.crop_size).permute(1, 0, 2, 3)

            gt_dot_return = gt_dot.view(num_h, self.crop_size, width).view(num_h, self.crop_size, num_w,
                                                                                   self.crop_size)
            gt_dot_return = gt_dot_return.permute(0, 2, 1, 3).contiguous().view(num_w * num_h, 1, self.crop_size,
                                                                                self.crop_size)

            return img_return, gt_dot_return

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for current_path in path:
            for maindir, subdir, file_name_list in os.walk(current_path):
                for filename in file_name_list:
                    if '_label' in filename:
                        continue
                    apath = os.path.join(maindir, filename)
                    ext = os.path.splitext(apath)[1]
                    if ext in image_ext:
                        image_paths.append(apath)
        return self.natural_sort(image_paths)
    
    def create_label_coordinates(self, dataPath, shape=(768,768)):
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
    
    def caculate_knn_distance(self, gt_points, num_point):

        if num_point >= 4:
            tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
            distances, locations = tree.query(gt_points, k=min(self.num_knn, num_point))
            distances = np.delete(distances, 0, axis=1)
            distances = np.mean(distances, axis=1)
            distances = torch.from_numpy(distances).unsqueeze(1)

        elif num_point == 0:
            distances = gt_points.clone()[:, 0].unsqueeze(1)

        elif num_point == 1:
            tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
            distances, locations = tree.query(gt_points, k=num_point)
            distances = torch.from_numpy(distances).unsqueeze(1)

        elif num_point == 2:
            tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
            distances, locations = tree.query(gt_points, k=num_point)
            distances = np.delete(distances, 0, axis=1)
            distances = (distances[:, 0]) / 1.0
            distances = torch.from_numpy(distances).unsqueeze(1)

        elif num_point == 3:
            tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
            distances, locations = tree.query(gt_points, k=num_point)
            distances = np.delete(distances, 0, axis=1)
            distances = (distances[:, 0] + distances[:, 1]) / 2
            distances = torch.from_numpy(distances).unsqueeze(1)

        return distances

class DataRandomCrop(Dataset):
    def __init__(self, data_path, ch, anydepth, augmentation, train, crop_size=256):
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.anydepth = anydepth
        self.augmentation = augmentation
        self.Counter = 0
        self.train = train
        self.crop_size = crop_size
        if self.channel == -2:
            REFERENCE_PATH = '/home/ocaki13/UNet-Torch/color_normalizer.npy'
            REF = np.load(REFERENCE_PATH)

            self.NORMALIZER = staintools.StainNormalizer(method='macenko')
            self.NORMALIZER.fit(REF)
        
    def transform(self, sample):
        image, label, gt_dot_map = sample['image'], sample['label'], sample['gt_dot']
        if self.augmentation:
            sample_list = [image, label, gt_dot_map]
            if random.random() > 0.5:
                sample_list = random_rot_flip(sample_list)
            elif random.random() > 0.5:
                sample_list = random_rotate(sample_list)
            image, label, gt_dot_map = sample_list
            
            
        #z normalizization
        mean3d = np.mean(image, axis=(0,1))
        std3d = np.std(image, axis=(0,1))
        image = (image-mean3d)/std3d
        if len(image.shape)==2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:  
            # HWC to CHW, BGR to RGB (for three channel)
            image = image.transpose((2, 0, 1))[::-1]
            image = torch.from_numpy(image.astype(np.float32))

        #image = self.normalizeTorch(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'gt_dot':gt_dot_map}
        return sample
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        if self.channel==1:
            if self.anydepth: 
                image = cv2.imread(img_path, -1)
            else:
                image = cv2.imread(img_path, 0)
        elif self.channel==3:
            image = cv2.imread(img_path)
        elif self.channel==-1:
            #image = cv2.imread(img_path)
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            rihc_hed = rgb2hed(im_rgb)
            image = rihc_hed[:,:,0]
        elif self.channel==-2:
            im_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            image = self.NORMALIZER.transform(im_rgb)

        label_path =  img_path.replace('.png', '_label.png')
        gt_path =  img_path.replace('.png', '_gt_dot.png') 

        label = cv2.imread(label_path, 0)
        gt_dot = cv2.imread(gt_path, 0)
        
        if self.train:
            img_cropped, label_cropped, gt_dot_cropped = self.crop(image, label, gt_dot)
            sample = {'image': img_cropped, 'label': label_cropped, 'gt_dot':gt_dot_cropped}
            sample = self.transform(sample)

            return sample['image'], sample['label'], sample['gt_dot']
        else:
            padding_h = image.shape[0] % self.crop_size
            padding_w = image.shape[1] % self.crop_size


            if padding_w != 0:
                padding_w = self.crop_size - padding_w
            if padding_h != 0:
                padding_h = self.crop_size - padding_h

            image_pad, label_pad, gt_dot_pad = pad_image([image, label, gt_dot], padding_w, padding_h)


            sample = {'image': image_pad, 'label': label_pad, 'gt_dot':gt_dot_pad}
            sample = self.transform(sample)
            image_pad = sample['image']
            label_pad = sample['label']
            gt_dot_pad = sample['gt_dot']

            a = []
            b = []
            c = []
            for i in range(0, image_pad.shape[1], self.crop_size):
                for j in range(0, image_pad.shape[2], self.crop_size):
                    a.append(image_pad[:,i:i+self.crop_size, j:j+self.crop_size])
                    b.append(label_pad[i:i+self.crop_size, j:j+self.crop_size])
                    c.append(gt_dot_pad[i:i+self.crop_size, j:j+self.crop_size])

            image_arr = torch.stack(a, axis=0)  # Shape: (N, 2, 256, 256)
            label_arr = torch.stack(b, axis=0)  # Shape: (N, 256, 256)
            gt_dot_arr = np.stack(c, axis=0)  # Shape: (N, 256, 256)
            
            return image_arr, label_arr, gt_dot_arr
            
            

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for current_path in path:
            for maindir, subdir, file_name_list in os.walk(current_path):
                for filename in file_name_list:
                    if '_label' in filename or '_gt_dot' in filename:
                        continue
                    apath = os.path.join(maindir, filename)
                    ext = os.path.splitext(apath)[1]
                    if ext in image_ext:
                        image_paths.append(apath)
        return self.natural_sort(image_paths)
    
    def crop(self, img, label, gt_dot):
        
        crop_size_r = random.randint(0, img.shape[0] - self.crop_size)
        crop_size_c = random.randint(0, img.shape[1] - self.crop_size)
        '''crop image'''
        img_croped = img[crop_size_r: crop_size_r + self.crop_size, crop_size_c:crop_size_c + self.crop_size, :]
        '''crop label'''
        label_croped = label[crop_size_r: crop_size_r + self.crop_size, crop_size_c:crop_size_c + self.crop_size]
        '''crop gt_dot'''
        gt_dot_croped = gt_dot[crop_size_r: crop_size_r + self.crop_size, crop_size_c:crop_size_c + self.crop_size]
        return img_croped, label_croped, gt_dot_croped