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


class Data_Reg_Binary(Dataset):
    def __init__(self, data_path, ch=1, anydepth=False, input_size=(512, 512), augmentation=False):
        super(Data_Reg_Binary, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.augmentation = augmentation
        self.anydepth = anydepth
        self.height = input_size[0]
        self.width = input_size[1]

    def transform_mask(self, img, mask):

        # Normalized
        if self.channel == 1:
            img = (img - img.mean()) / img.std()
            # HW to CHW (for gray scale)
            img = np.expand_dims(img, 0)
            # numpy to torch tensor
            img = torch.as_tensor(img)

        elif self.channel == 3:
            img[:, :, 0] = (img[:, :, 0] - img[:, :, 0].mean()
                            ) / img[:, :, 0].std()
            img[:, :, 1] = (img[:, :, 1] - img[:, :, 1].mean()
                            ) / img[:, :, 1].std()
            img[:, :, 2] = (img[:, :, 2] - img[:, :, 2].mean()
                            ) / img[:, :, 2].std()
            # HWC to CHW, BGR to RGB (for three channel)
            img = img.transpose((2, 0, 1))[::-1]
            img = torch.from_numpy(img.copy())
        else:
            raise ValueError('channel must be 1 or 3')

        # for 0 - 255
        # convert tensor with normalizzation
        # gt_mask_bin = TF.to_tensor(gt_mask_bin)

        # for 0 - 1 -2
        mask = np.expand_dims(mask, 0)  # comment for multiclass
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return img, mask

    def __getitem__(self, index):
        # read image
        imgPath = self.image_list[index]
        if self.anydepth:
            img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)
        else:
            if self.channel == 1:
                img = cv2.imread(imgPath, 0)
            elif self.channel == 3:
                img = cv2.imread(imgPath)
            else:
                raise ValueError('channel must be 1 or 3')
        r = max(self.width, self.height) / max(img.shape[:2])  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=interp)

        # read target label mask
        gt_mask_path = imgPath[:imgPath.rfind('.')] + '_label.png'
        gt_mask_bin = cv2.imread(gt_mask_path, 0)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_mask_bin = cv2.resize(gt_mask_bin, (self.width, self.height),
                                     interpolation=interp)

        # Preprocess
        img, gt_mask_bin = self.transform_mask(img, gt_mask_bin)

        # read distance map
        gtPath_dist = imgPath[:imgPath.rfind('.')] + '_dist_label.png'
        gt_dist = cv2.imread(gtPath_dist, 0)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_dist = cv2.resize(gt_dist, (self.width, self.height),
                                 interpolation=interp)

        # preprocess
        gt_dist = TF.to_tensor(gt_dist)

        return img, gt_mask_bin, gt_dist

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


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']

#         if random.random() > 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             image, label = random_rotate(image, label)
#         x, y = image.shape
#         if x != self.output_size[0] or y != self.output_size[1]:
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
#             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        
        
        
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.float32))
#         sample = {'image': image, 'label': label.long()}
#         return sample



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
            REFERENCE_PATH = "/kuacc/users/ocaki13/hpc_run/workfolder/color_normalizer.npy"
            REF = np.load(REFERENCE_PATH)

            self.NORMALIZER = staintools.StainNormalizer(method='macenko')
            self.NORMALIZER.fit(REF)
        
    def transform(self, sample):
        image, label = sample['image'], sample['label']

        if self.augmentation:
            print('zamazingo')
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
        
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


class Data_Reg_Fourier1(Dataset):
    def __init__(self, data_path, ch=1, anydepth=False, input_size=(512, 512), augmentation=False):
        super(Data_Reg_Fourier1, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.anydepth = anydepth
        self.augmentation = augmentation
        self.height = input_size[0]
        self.width = input_size[1]

    def transform_mask(self, img, mask, fmap):

        # # Random horizontal flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # # Random vertical flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.vflip(image)
        #     mask = TF.vflip(mask)

        # # Random rotation
        # if random.random() and self.augmentation > 0.5:
        #     angle = random.randint(10, 350)
        #     image = TF.rotate(image, angle)
        #     mask = TF.rotate(mask, angle)

        # # Brightness
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_brightness(image, random.uniform(0.5, 1.0))

        # # Contrast
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_contrast(image, random.uniform(0.5, 1.5))

        # # Gamma
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.adjust_gamma(image, random.uniform(0.5, 1))

        # # Gaussian Blur
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.gaussian_blur(image, (3, 3))

        # Normalized
        if self.channel == 1:
            img = (img - img.mean()) / img.std()
            # HW to CHW (for gray scale)
            img = np.expand_dims(img, 0)
            # numpy to torch tensor
            img = torch.as_tensor(img)

        elif self.channel == 3:
            img[:, :, 0] = (img[:, :, 0] - img[:, :, 0].mean()
                            ) / img[:, :, 0].std()
            img[:, :, 1] = (img[:, :, 1] - img[:, :, 1].mean()
                            ) / img[:, :, 1].std()
            img[:, :, 2] = (img[:, :, 2] - img[:, :, 2].mean()
                            ) / img[:, :, 2].std()
            # HWC to CHW, BGR to RGB (for three channel)
            img = img.transpose((2, 0, 1))[::-1]
            img = torch.from_numpy(img.copy())
        else:
            raise ValueError('channel must be 1 or 3')

        # Normalized
        fmap = (fmap - fmap.mean()) / fmap.std()
        # HW to CHW (for gray scale)
        fmap = np.expand_dims(fmap, 0)
        # HWC to CHW, BGR to RGB (for three channel)
        # img = img.transpose((2, 0, 1))[::-1]
        fmap = torch.as_tensor(fmap)

        # for 0 - 255
        # convert tensor with normalizzation
        # gt_mask_bin = TF.to_tensor(gt_mask_bin)

        # for 0 - 1 -2
        mask = np.expand_dims(mask, 0)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return img, mask, fmap

    def __getitem__(self, index):

        # read image
        imgPath = self.image_list[index]
        if self.anydepth:
            img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)
        else:
            if self.channel == 1:
                img = cv2.imread(imgPath, 0)
            elif self.channel == 3:
                img = cv2.imread(imgPath)
            else:
                raise ValueError('channel must be 1 or 3')
        r = max(self.width, self.height) / max(img.shape[:2])  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=interp)

        # read target label mask
        gt_mask_path = imgPath[:imgPath.rfind('.')] + '_label.png'
        gt_mask_bin = cv2.imread(gt_mask_path, 0)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_mask_bin = cv2.resize(gt_mask_bin, (self.width, self.height),
                                     interpolation=interp)

        # read distance map
        gtPath_fmap1 = imgPath[:imgPath.rfind('.')] + '_center2.fdmap1'
        gt_fmap1 = np.loadtxt(gtPath_fmap1)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_dist = cv2.resize(gt_dist, (self.width, self.height),
                                 interpolation=interp)

        # Preprocess
        img, gt_mask_bin, gt_fmap1 = self.transform_mask(
            img, gt_mask_bin, gt_fmap1)

        return img, gt_mask_bin, gt_fmap1

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


class Data_Reg_Fourier1_2(Dataset):
    def __init__(self, data_path, ch=1, anydepth=False, input_size=(512, 512), augmentation=False):
        super(Data_Reg_Fourier1_2, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.anydepth = anydepth
        self.augmentation = augmentation
        self.height = input_size[0]
        self.width = input_size[1]

    def transform_mask(self, img, mask, fmap1, fmap2):

        # # Random horizontal flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # # Random vertical flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.vflip(image)
        #     mask = TF.vflip(mask)

        # # Random rotation
        # if random.random() and self.augmentation > 0.5:
        #     angle = random.randint(10, 350)
        #     image = TF.rotate(image, angle)
        #     mask = TF.rotate(mask, angle)

        # # Brightness
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_brightness(image, random.uniform(0.5, 1.0))

        # # Contrast
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_contrast(image, random.uniform(0.5, 1.5))

        # # Gamma
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.adjust_gamma(image, random.uniform(0.5, 1))

        # # Gaussian Blur
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.gaussian_blur(image, (3, 3))

        # Normalized
        if self.channel == 1:
            img = (img - img.mean()) / img.std()
            # HW to CHW (for gray scale)
            img = np.expand_dims(img, 0)
            # numpy to torch tensor
            img = torch.as_tensor(img)

        elif self.channel == 3:
            img[:, :, 0] = (img[:, :, 0] - img[:, :, 0].mean()
                            ) / img[:, :, 0].std()
            img[:, :, 1] = (img[:, :, 1] - img[:, :, 1].mean()
                            ) / img[:, :, 1].std()
            img[:, :, 2] = (img[:, :, 2] - img[:, :, 2].mean()
                            ) / img[:, :, 2].std()
            # HWC to CHW, BGR to RGB (for three channel)
            img = img.transpose((2, 0, 1))[::-1]
            img = torch.from_numpy(img.copy())
        else:
            raise ValueError('channel must be 1 or 3')

        # Normalized
        fmap1 = (fmap1 - fmap1.mean()) / fmap1.std()
        # HW to CHW (for gray scale)
        fmap1 = np.expand_dims(fmap1, 0)
        # HWC to CHW, BGR to RGB (for three channel)
        # img = img.transpose((2, 0, 1))[::-1]
        fmap1 = torch.as_tensor(fmap1)

        # Normalized
        fmap2 = (fmap2 - fmap2.mean()) / fmap2.std()
        # HW to CHW (for gray scale)
        fmap2 = np.expand_dims(fmap2, 0)
        # HWC to CHW, BGR to RGB (for three channel)
        # img = img.transpose((2, 0, 1))[::-1]
        fmap2 = torch.as_tensor(fmap2)

        # for 0 - 255
        # convert tensor with normalizzation
        # gt_mask_bin = TF.to_tensor(gt_mask_bin)

        # for 0 - 1 -2
        mask = np.expand_dims(mask, 0)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return img, mask, fmap1, fmap2

    def __getitem__(self, index):

        # read image
        imgPath = self.image_list[index]
        if self.anydepth:
            img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)
        else:
            if self.channel == 1:
                img = cv2.imread(imgPath, 0)
            elif self.channel == 3:
                img = cv2.imread(imgPath)
            else:
                # ratio
                raise ValueError('channel must be 1 or 3')
        r = max(self.width, self.height) / max(img.shape[:2])
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=interp)

        # read target label mask
        gt_mask_path = imgPath[:imgPath.rfind('.')] + '_label.png'
        gt_mask_bin = cv2.imread(gt_mask_path, 0)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_mask_bin = cv2.resize(gt_mask_bin, (self.width, self.height),
                                     interpolation=interp)

        # read distance map
        gtPath_fmap1 = imgPath[:imgPath.rfind('.')] + '_center2.fdmap1'
        gt_fmap1 = np.loadtxt(gtPath_fmap1)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_fmap1 = cv2.resize(gt_fmap1, (self.width, self.height),
                                  interpolation=interp)

        # read distance map
        gtPath_fmap2 = imgPath[:imgPath.rfind('.')] + '_center2.fdmap2'
        gt_fmap2 = np.loadtxt(gtPath_fmap2)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_fmap2 = cv2.resize(gt_fmap2, (self.width, self.height),
                                  interpolation=interp)

        # Preprocess
        img, gt_mask_bin, gt_fmap1, gt_fmap2 = self.transform_mask(
            img, gt_mask_bin, gt_fmap1, gt_fmap2)

        return img, gt_mask_bin, gt_fmap1, gt_fmap2

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
