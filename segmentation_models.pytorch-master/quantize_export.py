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

def calculate_overlap_metrics(gt, pred,eps=1e-5):
    output = pred.view(-1, )
    target = gt.view(-1, ).float()

    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN

    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = ( tp + eps) / ( tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
#     specificity = (tn + eps) / (tn + fp + eps)

    return pixel_acc, dice,iou, precision, recall

def calculate_overlap_metrics_post(gt, pred,eps=1e-5):
    output = pred.view(-1, )/255.
    target = gt.view(-1, )/255.


    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN

    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = ( tp + eps) / ( tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
#     specificity = (tn + eps) / (tn + fp + eps)

    return pixel_acc, dice,iou, precision, recall


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
# acc_meter = AverageMeter()
# # train_loss_meter = AverageMeter()
# dice_meter = AverageMeter()
# iou_meter = AverageMeter()
pixel_acc_infected_meter= AverageMeter()
dice_infected_meter= AverageMeter()
iou_infected_meter= AverageMeter()
precision_infected_meter= AverageMeter()
recall_infected_meter= AverageMeter()

pixel_acc_lungs_meter= AverageMeter()
dice_lungs_meter= AverageMeter()
iou_lungs_meter= AverageMeter()
precision_lungs_meter= AverageMeter()
recall_lungs_meter= AverageMeter()


precision_classification_meter = AverageMeter()
recall_classification_meter = AverageMeter()
f1_score_classification_meter = AverageMeter()

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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('device selected: ',device)
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
weights_path = '/kaggle/input/baseline/inceptionv4.ckpt'

# Paths where ONNX and OpenVINO IR models will be stored.
onnx_path = 'quantized_model_final.onnx'
# ir_path = 'model_final.xml'  

# Load model for converting to ONNX 
# model = Multitask_MobileV3Smal_LRASPP(MobileNetV3_Modified, SegHead, SegHead)
state_dict = torch.load(weights_path, map_location='cpu')
# load state dict to model
model.load_state_dict(state_dict)
model.eval()

# # Set up data loaders
# train_data = Covid('/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data')
test_data = Covid('/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data',mode='test' )
# test_data = Covid(r'D:\Quan\AIVN\chest\Infection Segmentation Data\Infection Segmentation Data',mode='test' )

# print(x[0][1].shape)
#     t.append(time.time()-t1)

# print(np.mean(t[1:]))
# train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=2)



    
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

######################
def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


print_size_of_model(model)

##################
print('########quantization########')
model.qconfig = torch.quantization.default_qconfig
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
print_size_of_model(model_static_quantized)


# ##################
# # export full model
# IMAGE_WIDTH = 256
# IMAGE_HEIGHT = 256
# weights_path = '/kaggle/input/sample_best.ckpt'

# # Paths where ONNX and OpenVINO IR models will be stored.
# onnx_path = 'model_final.onnx'
# # ir_path = 'model_final.xml'  

# # Load model for converting to ONNX 
# # model = Multitask_MobileV3Smal_LRASPP(MobileNetV3_Modified, SegHead, SegHead)
# state_dict = torch.load(weights_path, map_location='cpu')
# # load state dict to model
# model.load_state_dict(state_dict)
# model.eval()
# print("Loaded PyTorch model")


# # Export model to ONNX
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore")
#     dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
#     input_names = [ "input_image" ]
#     output_names = [ "class", "lung_output", 'infect_output' ]

#     torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=input_names, output_names=output_names)
#     print(f"ONNX model exported to {onnx_path}.")
    

##################
# export quantizated model

# print("Loaded PyTorch model")


# Export model to ONNX
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    input_names = [ "input_image" ]
    output_names = [ "class", "lung_output", 'infect_output' ]

    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=input_names, output_names=output_names)
    print(f"ONNX model exported to {onnx_path}.")

#############################################

#############################################
from openvino.runtime import Core

print('evaluate')
# Instantiate OpenVINO Core
core = Core()

# Read model to OpenVINO Runtime
model_onnx = core.read_model(model=onnx_path)
# Load model on device
compiled_model_onnx = core.compile_model(model=model_onnx, device_name='CPU')
#############################################

# # res_onnx = compiled_model_onnx([normalized_input_image])[0]

# #########

# test_data = Covid('/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data',mode='test' )

# # print(x[0][1].shape)
# #     t.append(time.time()-t1)

# # print(np.mean(t[1:]))
# # train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
# val_loader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)


# for batch_idx, (inputs, labels_classification,  labels_segmentation_lungs, labels_segmentation_infected) in enumerate(val_loader):
#          # To device
#             inputs = inputs.to(device)
#             labels_classification = labels_classification.to(device)
#             labels_segmentation_infected = labels_segmentation_infected.to(device)
#             labels_segmentation_lungs = labels_segmentation_lungs.to(device)
            
#             outputs_classification, outputs_segmentation_lungs, outputs_segmentation_infected = model(inputs)
            
            
#             outputs_classification = outputs_classification.type(torch.float32)
#             outputs_segmentation_infected = outputs_segmentation_infected.type(torch.float32)
#             outputs_segmentation_lungs = outputs_segmentation_lungs.type(torch.float32)
            
#             labels_classification = labels_classification.type(torch.float32)
#             labels_segmentation_infected = labels_segmentation_infected.type(torch.float32)
#             labels_segmentation_lungs = labels_segmentation_lungs.type(torch.float32)
            
# #             print(outputs_classification ,labels_classification)
            
#     #         loss_classification = classification_loss_fn(outputs_classification, labels_classification)
#     #         loss_segmentation_infected = segmentation_loss_fn(outputs_segmentation_infected, labels_segmentation_infected)
#     #         loss_segmentation_lungs = segmentation_loss_fn(outputs_segmentation_lungs, labels_segmentation_lungs)
#     # #         loss = (1/3 * loss_classification) + (1/3 * loss_segmentation_infected) + (1/3 * loss_segmentation_lungs)
#     #         loss = (1/3 * loss_classification) + (1/3 * loss_segmentation_infected) + (1/3 * loss_segmentation_lungs)
#     #         val_loss += loss.item() * inputs.size(0)

#             outputs_classification = outputs_classification.argmax(1).detach().cpu().numpy()
#             outputs_segmentation_infected = outputs_segmentation_infected.argmax(1)
#             outputs_segmentation_lungs = outputs_segmentation_lungs.argmax(1)
            
#             labels_classification = labels_classification.argmax(1).detach().cpu().numpy()
#             labels_segmentation_infected = labels_segmentation_infected.argmax(1)
#             labels_segmentation_lungs = labels_segmentation_lungs.argmax(1)
                
            
#             pixel_acc_infected, dice_infected,iou_infected, precision_infected, recall_infected = calculate_overlap_metrics(labels_segmentation_infected, outputs_segmentation_infected,eps=1e-5)
#             pixel_acc_lungs, dice_lungs,iou_lungs, precision_lungs, recall_lungs = calculate_overlap_metrics(labels_segmentation_lungs, outputs_segmentation_lungs,eps=1e-5)
#             precision_classification = precision_score(labels_classification,outputs_classification,average='macro')
#             recall_classification = recall_score(labels_classification,outputs_classification,average='macro')
#             f1_score_classification = f1_score(labels_classification,outputs_classification,average='macro')
            
#             pixel_acc_infected_meter.update(pixel_acc_infected,inputs.shape[0])
#             dice_infected_meter.update(dice_infected,inputs.shape[0])
#             iou_infected_meter.update(iou_infected,inputs.shape[0])
#             precision_infected_meter.update(precision_infected,inputs.shape[0])
#             recall_infected_meter.update(recall_infected,inputs.shape[0])

#             pixel_acc_lungs_meter.update(pixel_acc_lungs,inputs.shape[0])
#             dice_lungs_meter.update(dice_lungs,inputs.shape[0])
#             iou_lungs_meter.update(iou_lungs,inputs.shape[0])
#             precision_lungs_meter.update(precision_lungs,inputs.shape[0])
#             recall_lungs_meter.update(recall_lungs,inputs.shape[0])

#             precision_classification_meter.update(precision_classification,inputs.shape[0])
#             recall_classification_meter.update(recall_classification,inputs.shape[0])
#             f1_score_classification_meter.update(f1_score_classification,inputs.shape[0])
# #             f1_score(y_true, y_pred, average='macro')
    

# print(f'pixel_acc_infected: {pixel_acc_infected_meter.avg :.4f}, dice_infected: {dice_infected_meter.avg :.4f},iou_infected: {iou_infected_meter.avg :.4f}, precision_infected: {precision_infected_meter.avg :.4f}, recall_infected: {recall_infected_meter.avg :.4f} \n \
# pixel_acc_lungs: {pixel_acc_lungs_meter.avg :.4f}, dice_lungs: {dice_lungs_meter.avg :.4f},iou_lungs: {iou_lungs_meter.avg :.4f}, precision_lungs: {precision_lungs_meter.avg :.4f}, recall_lungs: {recall_lungs_meter.avg :.4f} \n\
#     precision_classification: {precision_classification_meter.avg :.4f}, recall_classification: {recall_classification_meter.avg :.4f},f1_score_classification: {f1_score_classification_meter.avg :.4f} \n')
    
# ########

# print('valuate')
# def noise_remove(im):
#     kernel = np.ones((5, 5), np.uint8)
#     im_re = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel) 
#     contours, hierarchy = cv2.findContours(im_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # calculate points for each contour

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area <10:
#             cv2.fillPoly(im_re, pts =[cnt], color=(0))
#     return im_re


# def post_processing(outputs_classification, output_lungs, output_infected):
    
#     output_infected = noise_remove(output_infected)
#     output_lungs = noise_remove(output_lungs)
#     output_infected = cv2.bitwise_and(output_infected,output_lungs, mask = None)
            
#     return outputs_classification, output_lungs, output_infected
# elapsed_time = []
# with torch.no_grad():
#     for i in tqdm(range(len(test_data))):
#         input, labels_classification,  labels_segmentation_lungs, labels_segmentation_infected = test_data[i]
#         inputs = input.unsqueeze(0).to(device)
#         labels_classification = labels_classification.unsqueeze(0)
#         labels_segmentation_lungs = labels_segmentation_lungs.unsqueeze(0)
#         labels_segmentation_infected =labels_segmentation_infected.unsqueeze(0)
#         # inputs = input.unsqueeze(0)

#     # for batch_idx, (inputs, labels_classification,  labels_segmentation_lungs, labels_segmentation_infected) in enumerate(val_loader):
#     #     # To device
#     #     print('ss')
#         # inputs = inputs.to(device)
#         labels_classification = labels_classification.to(device)
#         labels_segmentation_infected = labels_segmentation_infected.to(device)
#         labels_segmentation_lungs = labels_segmentation_lungs.to(device)
        
#         # outputs_classification, outputs_segmentation_lungs, outputs_segmentation_infected = model(inputs)
#         start = time.time()
#         output_class, output_seg_lungs, output_seg_infected = compiled_model_onnx(inputs).values()
#         end = time.time()
#         outputs_segmentation_lungs = (np.transpose(outputs_segmentation_lungs.argmax(1).detach().cpu().numpy(), (1, 2, 0))*255).astype('uint8')
#         outputs_segmentation_infected = (np.transpose(outputs_segmentation_infected.argmax(1).detach().cpu().numpy(), (1, 2, 0))*255).astype('uint8')
#         # print
#         _,outputs_segmentation_lungs,outputs_segmentation_infected = post_processing(outputs_classification, outputs_segmentation_lungs, outputs_segmentation_infected)
#         elapsed_time.append(end - start)
       
#         outputs_segmentation_lungs = np.expand_dims(outputs_segmentation_lungs,axis=2)
#         outputs_segmentation_infected = np.expand_dims(outputs_segmentation_infected,axis=2)

#         # import matplotlib.pyplot as plt

#         # print(np.unique(outputs_segmentation_lungs))
#         # plt.imshow(outputs_segmentation_lungs,cmap='gray')
#         # cv2.imwrite('outputs_segmentation_lungs.jpg',outputs_segmentation_lungs)

#         outputs_classification = outputs_classification.argmax(1).detach().cpu().numpy()
#         # outputs_segmentation_infected = outputs_segmentation_infected.argmax(1)
#         # outputs_segmentation_lungs = outputs_segmentation_lungs.argmax(1)

#         labels_classification = labels_classification.argmax(1).detach().cpu().numpy()
#         labels_segmentation_infected = (np.transpose(labels_segmentation_infected.argmax(1).detach().cpu().numpy(), (1, 2, 0))*255).astype('uint8')
#         labels_segmentation_lungs = (np.transpose(labels_segmentation_lungs.argmax(1).detach().cpu().numpy(), (1, 2, 0))*255).astype('uint8')
#         # print(np.unique(labels_segmentation_lungs))

#         # cv2.imwrite('labels_segmentation_lungs.jpg',labels_segmentation_lungs)

        
        
#         pixel_acc_infected, dice_infected,iou_infected, precision_infected, recall_infected = calculate_overlap_metrics_post(torch.from_numpy(labels_segmentation_infected),torch.from_numpy(outputs_segmentation_infected),eps=1e-5)
#         pixel_acc_lungs, dice_lungs,iou_lungs, precision_lungs, recall_lungs = calculate_overlap_metrics_post(torch.from_numpy(labels_segmentation_lungs),torch.from_numpy( outputs_segmentation_lungs),eps=1e-5)
#         # print(dice_lungs,iou_lungs, precision_lungs, recall_lungs)
#         # break
        
#         precision_classification = precision_score(labels_classification,outputs_classification,average='macro')
#         recall_classification = recall_score(labels_classification,outputs_classification,average='macro')
#         f1_score_classification = f1_score(labels_classification,outputs_classification,average='macro')
        
#         pixel_acc_infected_meter.update(pixel_acc_infected,inputs.shape[0])
#         dice_infected_meter.update(dice_infected,inputs.shape[0])
#         iou_infected_meter.update(iou_infected,inputs.shape[0])
#         precision_infected_meter.update(precision_infected,inputs.shape[0])
#         recall_infected_meter.update(recall_infected,inputs.shape[0])

#         pixel_acc_lungs_meter.update(pixel_acc_lungs,inputs.shape[0])
#         dice_lungs_meter.update(dice_lungs,inputs.shape[0])
#         iou_lungs_meter.update(iou_lungs,inputs.shape[0])
#         precision_lungs_meter.update(precision_lungs,inputs.shape[0])
#         recall_lungs_meter.update(recall_lungs,inputs.shape[0])

#         precision_classification_meter.update(precision_classification,inputs.shape[0])
#         recall_classification_meter.update(recall_classification,inputs.shape[0])
#         f1_score_classification_meter.update(f1_score_classification,inputs.shape[0])

# #             f1_score(y_true, y_pred, average='macro')
# # val_loss /= len(val_loader.dataset)
# # scheduler.step(val_loss)
# # print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} \n \
# print(f'pixel_acc_infected: {pixel_acc_infected_meter.avg :.4f}, dice_infected: {dice_infected_meter.avg :.4f},iou_infected: {iou_infected_meter.avg :.4f}, precision_infected: {precision_infected_meter.avg :.4f}, recall_infected: {recall_infected_meter.avg :.4f} \n \
# pixel_acc_lungs: {pixel_acc_lungs_meter.avg :.4f}, dice_lungs: {dice_lungs_meter.avg :.4f},iou_lungs: {iou_lungs_meter.avg :.4f}, precision_lungs: {precision_lungs_meter.avg :.4f}, recall_lungs: {recall_lungs_meter.avg :.4f} \n\
#     precision_classification: {precision_classification_meter.avg :.4f}, recall_classification: {recall_classification_meter.avg :.4f},f1_score_classification: {f1_score_classification_meter.avg :.4f} \n')

# print(np.mean(elapsed_time[1:]))

#####################
print('inference')

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
# # Run inference on the input image
test_data = Covid('/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data', mode='test' )
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

# inv_tensor = invTrans(inp_tensor)
time_ls=[]
for i in range(580, 590):
    image, label_class, label_seg_lungs, label_seg_infected = test_data[i]
    

    fig, axs = plt.subplots(2, 3, figsize=(64,64))

    axs[0,0].imshow(invTrans(image).permute(1, 2, 0), cmap='gray')
    axs[0,0].set_title("Input image",fontsize = 20)
    axs[0,0].axis('off')

    axs[0,1].imshow(label_seg_lungs.argmax(0, keepdim=True).permute(1, 2, 0), cmap='gray')
    axs[0,1].set_title('Lung groundtruth',fontsize = 20)
    axs[0,1].axis('off')

    axs[0,2].imshow(label_seg_infected.argmax(0, keepdim=True).permute(1, 2, 0), cmap='gray')
    axs[0,2].set_title('Infected groundtruth',fontsize = 20)
    axs[0,2].axis('off')




    # fig = plt.figure(figsize=(20, 20))
    # fig.add_subplot(3, 3, 1)
    # fig.add_subplot(3, 3, 1).set_title('Image')
    # plt.imshow(invTrans(image).permute(1, 2, 0), cmap='gray')
    # plt.axis('off')

    # fig.add_subplot(3, 3, 2)
    # fig.add_subplot(3, 3, 2).set_title('Lung groundtruth')
    # plt.imshow(label_seg_lungs.argmax(0, keepdim=True).permute(1, 2, 0), cmap='gray')
    # plt.axis('off')

    # fig.add_subplot(3, 3, 3)
    # fig.add_subplot(3, 3, 3).set_title('Infected Lung groundtruth')
    # plt.imshow(label_seg_infected.argmax(0, keepdim=True).permute(1, 2, 0), cmap='gray')
    # plt.axis('off')

    
    image = image.unsqueeze(0).to('cpu').numpy()
    with torch.no_grad():
        t1= time.time()
        output_class, output_seg_lungs, output_seg_infected = compiled_model_onnx(image).values()

    output_class = output_class.argmax(1)
    output_seg_lungs = (np.transpose(output_seg_lungs.argmax(1), (1, 2, 0))*255).astype('uint8')
    output_seg_infected = (np.transpose(output_seg_infected.argmax(1), (1, 2, 0))*255).astype('uint8')
      
    _, output_seg_lungs, output_seg_infected, infected_ratio, illustrate_im = post_processing_inf(output_class, output_seg_lungs, output_seg_infected)
    t2= time.time()-t1
    time_ls.append(t2)

    axs[1,0].imshow(output_seg_lungs,cmap='gray')
    axs[1,0].set_title('Lung output',fontsize = 20)
    axs[1,0].axis('off')

    axs[1,1].imshow(output_seg_infected,cmap='gray')
    axs[1,1].set_title('Infected output',fontsize = 20)
    axs[1,1].axis('off')

    axs[1,2].imshow(illustrate_im,cmap='gray')
    axs[1,2].set_title('Final output',fontsize = 20)
    axs[1,2].axis('off')

    # fig.add_subplot(3, 3, 4)
    # fig.add_subplot(3, 3, 4).set_title('Lung output')
    # plt.imshow(output_seg_lungs,cmap='gray')
    # plt.axis('off')


    # fig.add_subplot(3, 3, 5)
    # fig.add_subplot(3, 3, 5).set_title('Infected Lung output')
    # plt.imshow(output_seg_infected,cmap='gray')  
    # plt.axis('off')

    # fig.add_subplot(3, 3, 6)
    # fig.add_subplot(3, 3, 6).set_title('Final output')
    # plt.imshow(illustrate_im,cmap='gray')
    # plt.axis('off')

    plt.savefig('/kaggle/working/asfsaf.png')
print(np.mean(time_ls[1:]))