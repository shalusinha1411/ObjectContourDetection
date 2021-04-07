# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
import os
from PIL import Image
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
import cv2
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt


class BSDS(Dataset):
    def __init__(self, rootDirImg, rootDirGt, processed=True):
        self.rootDirImg = rootDirImg
        self.rootDirGt = rootDirGt
        self.listData = [sorted(os.listdir(rootDirImg)),sorted(os.listdir(rootDirGt))]
        self.processed = processed

    def __len__(self):
        return len(self.listData[1])
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[0][i]
        targetName = self.listData[1][i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        targetImage = transf(Image.open(self.rootDirGt + targetName).convert('L'))
        targetImage = (targetImage>0.41).float()
        return inputImage, targetImage

    def preprocess(self):
        if self.processed:
            return

        for targetName in self.listData[1]:
            targetFile = loadmat(self.rootDirGt + targetName)['groundTruth'][0]
            s = (len(targetFile[0][0][0][1]), len(targetFile[0][0][0][1][0]))
            result = np.zeros(s)
            num = len(targetFile)
            for target in targetFile:
                element = target[0][0][1]
                result = np.add(result, element)
            
            result = result/num
            # save result as png image
            result = (result*255.0).astype(np.uint8)
            img = Image.fromarray(result, 'L')
            targetSaveName = targetName.replace(".mat", ".png")
            img.save(self.rootDirGt + targetSaveName)
        # erase all .mat files
        os.system("rm " + self.rootDirGt + "*.mat")
        # update dataset
        self.listData = [sorted(os.listdir(self.rootDirImg)),sorted(os.listdir(self.rootDirGt))]
        self.processed = True

class BSDS_TEST(Dataset):
    def __init__(self, rootDirImg):
        self.rootDirImg = rootDirImg
        self.listData = sorted(os.listdir(rootDirImg))

    def __len__(self):
        return len(self.listData)
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        inputName = inputName.split(".jpg")[0] + ".png"
        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434
        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)
        return inputImage, inputName

class TrainDataset(Dataset):
    def __init__(self, fileNames, rootDir):        
        self.rootDir = rootDir
        self.transform = transforms.ToTensor()
        self.targetTransform = transforms.ToTensor()
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0])
        targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1])

        # process the images
        inputImage = Image.open(inputName).convert('RGB')
        inputImage = self.transform(inputImage)

        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434

        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)

        targetImage = Image.open(targetName).convert('L')
        targetImage = self.targetTransform(targetImage)
        targetImage = (targetImage>0.41).float()
        return inputImage, targetImage

class bsdsdataloader:
    def __init__(self,paths):
        self.paths = paths
        self.train_images_pil = []
        self.train_tar_pil = []
        self.make_train_val_names()
        self.pointer = 0
        self.index_arr = [x for x in range(len(self.train_names))]

        
    def make_train_val_names(self):
        with open(paths["train_names_path"]) as handle:
            orignal_train_names = [x.split("\n")[0].strip() for x in handle]
        self.train_names = []
        classaug_imgs = os.listdir(self.paths["targets_path"])
        for name in classaug_imgs:
            name = name.split(".")[0]
        self.train_names += orignal_train_names

        self.train_names = list(set(self.train_names))


    def __len__(self):
        return len(self.train_names)

    def reset_loader(self):
        random.shuffle(self.index_arr)
        self.pointer = 0

    def transform(self, image_origin, mask_origin, mode, data_augmentation = "randomcrop"):
        image_res, mask_res = None, None
        totensor_op = transforms.ToTensor()
        color_op = transforms.ColorJitter(0.1, 0.1, 0.1)
        resize_op = transforms.Resize((224, 224))
        image_origin = color_op(image_origin)
      #  norm_op = transforms.Normalize(mean =[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        if mode == 'val' or mode == 'predict':
            image_res = totensor_op(image_origin)
            mask_res = totensor_op(mask_origin)
        elif mode == 'train':
            if data_augmentation == 'randomcrop':
                if image_origin.size[0] < 224 or image_origin.size[1] < 224:
                    #padding-val:
                    val = int(np.array(image_origin).sum() / image_origin.size[0] / image_origin.size[1])
                    padding_width = 224-min(image_origin.size[0],image_origin.size[1])
                    padding_op = transforms.Pad(padding_width,fill=val)
                    image_origin = padding_op(image_origin)
                    padding_op = transforms.Pad(padding_width, fill=0)
                    mask_origin = padding_op(mask_origin)
                i, j, h, w = transforms.RandomCrop.get_params(
                    image_origin, output_size=(224, 224)
                )
                image_res = totensor_op(tvf.crop(image_origin, i, j, h, w))
                mask_res = totensor_op(tvf.crop(mask_origin, i, j, h, w))

            elif data_augmentation == 'resize':
                image_res = totensor_op(resize_op(image_origin))
                mask_res = totensor_op(resize_op(mask_origin))
      #  image_res = norm_op(image_res)
     
        return image_res, mask_res


    def _make_dicts(self):
        count= 0
        for img_name in self.train_names:
            count += 1
            print(count)
            img_path = os.path.join(self.paths["images_path"] , img_name + ".jpg")
            mask_path = os.path.join(self.paths["targets_path"] , img_name + ".png")
            img = Image.open(img_path)
            tar = Image.open(mask_path)
            self.train_images_pil.append(img)
            self.train_tar_pil.append(tar)
        
    
    
    def get_next_mini_batch(self, index_given= None):
            if(self.pointer == len(self.train_names)):
                self.reset_loader()
            if(index_given == None):
                index = self.pointer
            else:
                index = index_given
            self.pointer += 1
            img_name = self.train_names[index]
            img_path = os.path.join(self.paths["images_path"] , img_name + ".jpg")
            mask_path = os.path.join(self.paths["targets_path"] , img_name + ".png")
            print(mask_path)
            img = Image.open(img_path)
            tar = cv2.imread(mask_path,0)
            tar = Image.fromarray(tar)
            img_batch = torch.empty(8,3,224,224)
            tar_batch = torch.empty(8,1,224,224)

            for i in range(4):
                img_batch[i] , tar_batch[i] = self.transform(img,tar,"train")
            img = tvf.hflip(img)
            tar = tvf.hflip(tar)

            for i in range(4,8):
                img_batch[i] , tar_batch[i] = self.transform(img,tar,"train")

           # tar_batch[tar_batch > .9] = 1
           # tar_batch[tar_batch <= .9] = 0
            return img_batch , tar_batch
            
            
    def get_next_batch(self, batch_size):
        index = np.random.randint(0, len(self.train_names), size = batch_size)
        img_batch = torch.empty(batch_size,3,224,224)
        tar_batch = torch.empty(batch_size,1,224,224)

        for i in range(batch_size):
            img_name = self.train_names[index[i]]
            img_path = os.path.join(self.paths["images_path"] , img_name + ".jpg")
            mask_path = os.path.join(self.paths["targets_path"] , img_name + ".png")
            img = Image.open(img_path)
            tar = cv2.imread(mask_path,0)
            tar = Image.fromarray(tar)
            
            if(random.random() > .5):
                img = tvf.hflip(img)
                tar = tvf.hflip(tar)
            img_batch[i] , tar_batch[i] =  self.transform(img,tar,"train")

        return img_batch, tar_batch
