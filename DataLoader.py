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

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png', '.tif', '.PNG', '.tiff']


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
        
    def transform(self, sample):
        label1, label2 = sample['label']
        image = sample['image'] 

        if self.augmentation:
            sample_list = [image, label1, label2]
            if random.random() > 0.5:
                sample_list = random_rot_flip(sample_list)
            elif random.random() > 0.5:
                sample_list = random_rotate(sample_list)
            image, label1, label2 = sample_list
        
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
        label_immune = np.load(label_path_immune)
        label_other = np.load(label_path_other)
       
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
            sample_list = [image, label]
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

        label_path =  img_path[:img_path.rfind('.')] + '_label.png'
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
