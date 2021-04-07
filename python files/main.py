import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
import torch
from torchvision.transforms import functional as tvf
import torchvision
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
import scipy.ndimage.filters
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue 
import time
import torch.nn.functional as F
import sys
from net import get_model
from train import dataloader, trainer, multiprocess_control
from preprocessing import bsdsdataloader, BSDS, BSDS_TEST, TrainDataset
from eval import Evaluator

#VOC2012 DATASET
vocpaths = {
"images_path" : "./JPEGImages",
"targets_path" : "./SegmentationObjectFilledDenseCRF",
"train_names_path" : "./train.txt",
"val_names_path" : "./val.txt",
"new_contours_path": "./improved_contours"
}

for p in vocpaths.values():
    if(os.path.exists(p) == False):
        print("path " , p , "does not exist")

#training the model on VOC 2012
if __name__ == "__main__":
    assert len(sys.argv) <= 2 , "Enter the the model number to resume from"
    last_model = 0
    if(len(sys.argv) == 2):
        last_model  = sys.argv[1]
        model = get_model("model" + str(last_model) + ".pth")
    else:
        model = get_model()

    data_object = dataloader(vocpaths)
    
    T = trainer(model)

    loss_voc =[]
    multi = multiprocess_control(data_object,workers = 2,produce_func = producer, shared_Q = Q,epochs_desired = 30,train_object =T, offset = last_model,loss_voc)
    multi.start_training


#evaluating model trained on VOC 2012, generating PR curve, loss_vs_epochs curve, F_score and AP
vocevalpaths = {
"images_path" : "./JPEGImages",
"targets_path" : "./improved_contours",
"train_names_path" : "./train.txt",
"val_names_path" : "./val.txt",
"results_path" : "./results",
"models_path" : "./models",
"CRF_mask_path" :"./SegmentationObjectFilledDenseCRF"
}


targets_names = []

with open(paths["val_names_path"]) as handle:
    for name in handle:
        name = name.split("\n")[0].strip()
        targets_names.append(name)

targets_paths = []
results_paths = []
for name in targets_names:
    targets_path = os.path.join(paths["targets_path"], name + ".png")
    targets_paths.append(targets_path)

    result_path = os.path.join(paths["results_path"] , name + ".npy")
    results_paths.append(result_path)


if __name__ == "__main__":
        evl = Evaluator(targets_paths , results_paths)
        P, R, thres = evl.calc_PR_curve(300)
        plt.plot(R,P)
        plt.show()
        
        f_score = -1
        f_thres = -1
        index = 0
        for p,r,t in zip(P,R,thres.tolist()):
            if(p == 0 or r == 0):
                continue
            f1 = 2*p*r/(p + r)

            if(f1 > f_score ):
                f_score = f1
                f_thres = t
            index += 1
        print("max f score = " + str(f_score) + " at threshhold = " + str(f_thres/301))

        AP, MR, MP = evl.voc_ap(R,P)
        print(AP)

        plot_loss(loss_voc)



#BSDS500 DATASET
bsdspaths = {
"images_path" : "./BSDS500/train",
"targets_path" : "./BSDS500/groundTruth/train",
"train_names_path" : "./BSDS500/train.txt",
"model_save_path": "./models",
"BSDS500_test_path" : "./BSDS500/test"
}

for p in bsdspaths.values():
    if(os.path.exists(p) == False):
        print("path " , p , "does not exist")


#preprocess
rootDirImgTrain = "./BSDS500/train"
rootDirGtTrain = "./BSDS500/groundTruth/train/"
rootDirImgVal = "./BSDS500/val"
rootDirGtVal= "./BSDS500/groundTruth/val"
rootDirImgTest = "./BSDS500/test/"
rootDirGtTest = "./BSDS500/groundTruth/test/"

trainDS = BSDS(rootDirImgTrain, rootDirGtTrain, processed = False)
valDS = BSDS(rootDirImgVal, rootDirGtVal, processed = False)

trainDS.preprocess()
valDS.preprocess()

testDS = BSDS(rootDirImgTest, rootDirGtTest, processed = False)
testDS.preprocess()


#pre-trained model
model = countour_detector(vgg)
model.load_state_dict(torch.load(os.path.join(paths["model_save_path"],"model30.pth")))
model = model.to(device)
#evaluating pre-trained model

#fine-tuning the model for BSDS500       
if __name__ == "__main__":
    assert len(sys.argv) <= 2 , "Enter the the model number to resume from"
    last_model = 0
    if(len(sys.argv) == 2):
        last_model  = sys.argv[1]
        model = get_model("model" + str(last_model) + ".pth")
    else:
        model = get_model()

    data_object = bsdsdataloader(bsdspaths)
    
    T = trainer(model)
    T.lr = 1e-5
    loss_bsds_ft = []

    multi = multiprocess_control(data_object,workers = 2,produce_func = producer, shared_Q = Q,epochs_desired = 100,train_object =T, offset = last_model,loss_bsds_ft)
    multi.start_training()

#evaluating fine-tuned model
ftpaths = {
"images_path" : "./BSDS500/test",
"targets_path" : "./BSDS500/groundTruth/test",
"testBSDS_names_path" : "./BSDS500/test.txt",
"results_path" : "./BSDS500/results",
"model_save_path" : "./models"
}

targets_names = []

with open(paths["testBSDS_names_path"]) as handle:
    for name in handle:
        name = name.split("\n")[0].strip()
        targets_names.append(name)

targets_paths = []
results_paths = []

for img in val_names: 
   test_path = os.path.join(paths["images_path"], img + ".jpg")
   test_img = Image.open(test_path)
   test_img = np.array(test_img)
   test_img = np.rollaxis(test_img , 2)
   test_img = torch.tensor(test_img).unsqueeze(0).to(device).float()/255
   res = model(test_img)
   a = res.cpu().detach().numpy()[0][0]
   a[a >= 5] = 1
   a[a < .5] = 0
   np.save(os.path.join(paths["results_path"], img + ".npy"), a)
   print(img)
   print('saved')

if __name__ == "__main__":
        evl = Evaluator(targets_paths , results_paths)
        P, R, thres = evl.calc_PR_curve(300)
        plt.plot(R,P)
        plt.show()
        
        f_score = -1
        f_thres = -1
        index = 0
        for p,r,t in zip(P,R,thres.tolist()):
            if(p == 0 or r == 0):
                continue
            f1 = 2*p*r/(p + r)

            if(f1 > f_score ):
                f_score = f1
                f_thres = t
            index += 1
        print("max f score = " + str(f_score) + " at threshhold = " + str(f_thres/301))

        AP, MR, MP = evl.voc_ap(R,P)
        print(AP)

        #plot_loss(loss_bsds_ft )
