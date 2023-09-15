import segmentation_models_pytorch as smp
import torch

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    # activation='sigmoid',      # activation function, default is None
    classes=3,                 # define number of output labels
)
model = smp.Unet(
    encoder_name="inceptionv4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,  
    aux_params=aux_params                    # model output channels (number of classes in your dataset)
)


import warnings

import numpy as np
# import imgaug.augmenters as iaa
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import cv2
import torch.nn.functional as F
class Covid(Dataset):
    def __init__(self, rootpath, cropsize=(256, 256), mode='train', *args, **kwargs):
        super(Covid, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.cropsize=cropsize

        self.rootpth = rootpath
        self.imgs = []

        class_dir = os.listdir(os.path.join(self.rootpth,self.mode.capitalize()))  # class
        self.class_cvt = { 'Normal':0, 'COVID-19':1,'Non-COVID':2}
    
        # print(class_dir)
        for _class in class_dir:
            self.class_path = os.path.join(self.rootpth,self.mode.capitalize())
            self.img_path = os.path.join(self.class_path,_class,'images')
#             print(self.img_path)
            img_lst = os.listdir(self.img_path)
#             print(img_lst)
#             infect_imgs = os.listdir(self.img_path.replace('Lung Segmentation Data','Infection Segmentation Data',2))
            img_lst.sort()
            for img in img_lst:
                self.imgs.append((img,self.class_cvt[_class]))
#         print(self.imgs)


#         #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
#         self.trans_color = iaa.Sequential([
#           iaa.LinearContrast((0.4, 1.6)),
# #           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
# #           iaa.Add((-40, 40), per_channel=0.5, name="color-jitter")
#         ])
#         self.trans_train = iaa.Sequential([
# #           iaa.Resize(self.cropsize),
#           iaa.Fliplr(0.5),
# #           iaa.Affine(rotate=(-45, 45),
# #                     translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}),
#       ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
#         print(impth)
        self.img_path = os.path.join(self.class_path,list(self.class_cvt.keys())[impth[1]],'images')
        img =  cv2.imread(os.path.join(self.img_path,impth[0]))
        img = cv2.resize(img,self.cropsize, cv2.INTER_LINEAR)
        
#         label = torch.tensor(impth[1])
        label = F.one_hot(torch.tensor(impth[1]), num_classes=3)
#         print(label)

        mask_path = self.img_path.replace('images','lung masks')
        
        lung_mask = cv2.imread(os.path.join(mask_path,impth[0]),0)
        lung_mask = cv2.resize(lung_mask,self.cropsize, cv2.INTER_NEAREST ).astype(np.int64)
#         lung_mask = np.where(lung_mask==0, 0 , 1)
        
        infect_path = self.img_path.replace('images','infection masks')
        infect_mask = cv2.imread(os.path.join(infect_path,impth[0]),0)
        infect_mask = cv2.resize(infect_mask,self.cropsize, cv2.INTER_NEAREST ).astype(np.int64)
#         infect_mask = np.where(infect==0, 0 , 1)

        #if self.mode == 'test':
            #return img, label,lung_mask, infect_mask
            
            
        
        if self.mode == 'train':
#             color = self.trans_color.to_deterministic()

#             img = color.augment_image(img)
            det_tf = self.trans_train.to_deterministic()
            img = det_tf.augment_image(img)
            lung_mask = det_tf.augment_image( lung_mask)
            infect_mask = det_tf.augment_image(infect_mask)


        img = self.to_tensor(img)
#         img = img.permute(2, 0, 1)
        
        lung_mask = np.where(lung_mask !=0,1.,0.)
        lung_mask = torch.from_numpy(lung_mask.astype(np.float32)).clone()
        lung_mask = F.one_hot(lung_mask.long(), num_classes=2)
        lung_mask = lung_mask.to(torch.float32)
        lung_mask = lung_mask.permute(2, 0, 1)
        
        infect_mask = np.where(infect_mask !=0,1.,0.)
        infect_mask = torch.from_numpy(infect_mask.astype(np.float32)).clone()
        infect_mask = F.one_hot(infect_mask.long(), num_classes=2)
        infect_mask = infect_mask.to(torch.float32)
        infect_mask = infect_mask.permute(2, 0, 1)
        

        return img, label, lung_mask, infect_mask



    def __len__(self):
        return len(self.imgs)
    
import gc
torch.cuda.empty_cache()
gc.collect()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm.notebook import tqdm as tqdm
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision
import time
# learning_rate = 0.001
# num_epochs = 200

# # Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device selected: ',device)
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
weights_path = '/kaggle/input/baseline/inceptionv4.ckpt'

state_dict = torch.load(weights_path, map_location=device)

model.load_state_dict(state_dict)
model.to(device)
model.eval()


test_data = Covid('/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data',mode='test' )



print('after postprocess')
def noise_remove(im):
    kernel = np.ones((5, 5), np.uint8)
    im_re = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel) 
    contours, hierarchy = cv2.findContours(im_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# calculate points for each contour

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <10:
            cv2.fillPoly(im_re, pts =[cnt], color=(0))
    return im_re


def post_processing(outputs_classification, output_lungs, output_infected):
    output_infected = noise_remove(output_infected)
    output_lungs = noise_remove(output_lungs)
    output_infected = cv2.bitwise_and(output_infected,output_lungs, mask = None)
            
    return outputs_classification, output_lungs, output_infected

def post_processing_inf(outputs_classification, output_lungs, output_infected):
    class_revert_cvt = { 0:'Normal',1: 'COVID-19',2:'Non-COVID'}
    
    if outputs_classification.tolist()[0] == 1:
        output_infected = noise_remove(output_infected)
        output_lungs = noise_remove(output_lungs)
        illustrate_im = cv2.cvtColor(output_lungs.copy(),cv2.COLOR_GRAY2RGB)
        output_infected = cv2.bitwise_and(output_infected,output_lungs, mask = None)
        infected_ratio = 100*np.count_nonzero(output_infected)/(np.count_nonzero(output_lungs)+1e-5)
        outputs_classification = class_revert_cvt[outputs_classification.tolist()[0]]
        
        contours, hierarchy = cv2.findContours(output_infected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(illustrate_im, contours, -1, (0, 255, 0), 1)
        illustrate_im = cv2.putText(illustrate_im, f'Infected ratio: {infected_ratio:.4f}%',(5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0))
        illustrate_im = cv2.putText(illustrate_im, f'Predicted: {outputs_classification}',(5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0))
        
        return outputs_classification, output_lungs, output_infected, infected_ratio, illustrate_im
    else:
        output_infected = np.zeros_like(output_infected) 
        output_lungs = noise_remove(output_lungs)
        illustrate_im = cv2.cvtColor(output_lungs.copy(),cv2.COLOR_GRAY2RGB)
        outputs_classification = class_revert_cvt[outputs_classification.tolist()[0]]
        illustrate_im = cv2.putText(illustrate_im, f'Infected ratio: 0%',(5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0))
        illustrate_im = cv2.putText(illustrate_im, f'Predicted: {outputs_classification}',(5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0))
        return outputs_classification, output_lungs, output_infected, 0, illustrate_im
    

for i in range(len(test_data)):
    image, label_class, label_seg_lungs, label_seg_infected = test_data[i]
    inputs = image.unsqueeze(0).to(device)
    # labels_classification = labels_classification.to(device)
    # labels_segmentation_infected = labels_segmentation_infected.to(device)
    # labels_segmentation_lungs = labels_segmentation_lungs.to(device)
    
    output_class, output_seg_lungs, output_seg_infected = model(inputs)

    output_class = output_class.argmax(1)
    output_seg_lungs = (np.transpose(output_seg_lungs.argmax(1).detach().cpu().numpy(), (1, 2, 0))*255).astype('uint8')
    output_seg_infected = (np.transpose(output_seg_infected.argmax(1).detach().cpu().numpy(), (1, 2, 0))*255).astype('uint8')
    _, output_seg_lungs, output_seg_infected, infected_ratio, illustrate_im = post_processing_inf(output_class, output_seg_lungs, output_seg_infected)
    cv2.imwrite(f'/kaggle/working/Lung/{i}.png',output_seg_lungs )
    cv2.imwrite(f'/kaggle/working/Infected/{i}.png',output_seg_infected )
    # cv2.imwrite(f'/kaggle/working/illustrate_im.png',illustrate_im )