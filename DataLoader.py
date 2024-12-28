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
image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png', '.tif', '.PNG', '.tiff']



def PathologyAugmentationAug(sample, aug):
    image, label = sample['image'], sample['label']
    
    # Wrap the segmentation map
    segmap = SegmentationMapsOnImage(label, shape=image.shape)
    image_aug, segmap_aug = aug(images=[image], segmentation_maps=[segmap])
    
    return {'image': image_aug[0], 'label': segmap_aug[0].get_arr()}

def PathologyAugmentationAugHM(sample, aug):
    image = sample['image']
    label1, label2 = sample['label']

    # Wrap the heatmap(s)
    heatmaps_obj = HeatmapsOnImage(np.stack((label1, label2),axis=-1), shape=image.shape)
    
    # Apply the augmentation
    image_aug, heatmaps_aug = aug(images=[image], heatmaps=[heatmaps_obj])
    
    # Extract the augmented image and heatmap(s)
    return {'image': image_aug[0], 'label': [heatmaps_aug[0].get_arr()[:,:,0], heatmaps_aug[0].get_arr()[:,:,1]]}
    
    
    
# MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
#                     "Fliplr", "Flipud", "CropAndPad",
#                     "Affine", "PiecewiseAffine"]

# def hook(images, augmenter, parents, default):
    
#     """Determines which augmenters to apply to masks."""
#     return augmenter.__class__.__name__ in MASK_AUGMENTERS

# def PathologyAugmentationAug(sample, aug):
#     det = aug.to_deterministic()    
#     augmentedSample = {}
#     for currentSample in sample:
#         if 'label' in currentSample: 
#             augmentedSample[currentSample] = det.augment_image(sample[currentSample],
#                                  hooks=imgaug.HooksImages(activator=hook))
#             augmentedSample[currentSample][augmentedSample[currentSample]==255] = 0
#         else:
#             augmentedSample[currentSample] = det.augment_image(sample[currentSample])
#     return augmentedSample


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
        
    def transform(self, sample):
        image, label = sample['image'], sample['label']

        if self.augmentation:
            sample_list = [image,label]
            if random.random() > 0.5:
                sample_list = random_rot_flip(sample_list)
            elif random.random() > 0.5:
                sample_list = random_rotate(sample_list)
            image, label = sample_list
        
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

        #image = self.normalizeTorch(image.astype(np.float32))
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
        # label_path_immune =  img_path[:img_path.rfind('.')] + '_label_immune_reg.npy'
        # label_path_other =  img_path[:img_path.rfind('.')] + '_label_other_reg.npy'
        label = np.load(label_path)
        
        # print(np.sum(label))
        # print(label)
        sample = {'image': image, 'label': label}
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
                sample = PathologyAugmentationAugHM(sample, self.aug)
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
        
    def transform(self, sample):
        image, label = sample['image'], sample['label']
        self.Counter +=1
        if self.augmentation:
            # if random.random() > 0.5:
            #     sample = RadiologyAugmentationTIO(sample, self.transforms_dict)
            #     image, label = sample['image'], sample['label']
            #     cv2.imwrite(os.path.join('deneme/','torchio'+str(self.Counter)+'.png'),image)
            #     cv2.imwrite(os.path.join('deneme/','torchio'+str(self.Counter)+'_label.png'),label)
                
                
            if random.random() > 0.25:
                
                #image_org, label_org = sample['image'], sample['label']
                            
                sample = PathologyAugmentationAug(sample, self.aug)
                image, label = sample['image'], sample['label']
                image = np.array(self._colorJitter(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # ##### DEBUG ######            
                # # Create a vertical line as a separator
                # separator_width = 10
                # separator = np.zeros((image.shape[0], separator_width, 3), dtype=np.uint8) 

                # # Concatenate the images with the separator
                # concatenated_image = np.hstack((image_org, separator, image))
                # separator = np.zeros((image.shape[0], separator_width), dtype=np.uint8) 
                # concatenated_label = np.hstack((label_org, separator, label))

                # # Save the concatenated image
                # cv2.imwrite(os.path.join('augload/','imgaug'+str(self.Counter)+'aug.png'),concatenated_image)
                # cv2.imwrite(os.path.join('augload/','imgaug'+str(self.Counter)+'aug_label.png'),concatenated_label)


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

        #image = self.normalizeTorch(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
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

        label_path =  img_path[:img_path.rfind('.')] + '_label_mc.png'
        label = cv2.imread(label_path, 0)
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
