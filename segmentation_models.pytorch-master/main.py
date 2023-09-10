import segmentation_models_pytorch as smp
import torch
aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    # activation='sigmoid',      # activation function, default is None
    classes=3,                 # define number of output labels
)
model = smp.Unet(
    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,  
    aux_params=aux_params                    # model output channels (number of classes in your dataset)
)

a = 0.1 #class, infect
# print(model(torch.ones([1, 3, 256, 256]))[-1])
#data loader r
import numpy as np
import imgaug.augmenters as iaa
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
        self.trans_color = iaa.Sequential([
          iaa.LinearContrast((0.4, 1.6)),
#           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
#           iaa.Add((-40, 40), per_channel=0.5, name="color-jitter")
        ])
        self.trans_train = iaa.Sequential([
#           iaa.Resize(self.cropsize),
          iaa.Fliplr(0.5),
#           iaa.Affine(rotate=(-45, 45),
#                     translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}),
      ])

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
learning_rate = 0.001
num_epochs = 200

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device selected: ',device)
# Set up data loaders
train_data = Covid('/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data')
val_data = Covid('/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data',mode='val' )

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=2)
# Set up model
model = model.to(device)

# Set up loss function
classification_loss_fn = nn.CrossEntropyLoss()
# segmentation_loss_fn = torchvision.ops.sigmoid_focal_loss
segmentation_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# Set up optimizer
patience = 300
best_acc = 0
save_every = 5
# Set up training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    print(f">>> Training epoch {epoch}")
    progress = tqdm(train_loader, total=int(len(train_loader)))
    for batch_idx, (inputs, labels_classification, labels_segmentation_lungs, labels_segmentation_infected) in enumerate(progress):
    # Training
        progress.refresh()
         # To device
        inputs = inputs.to(device)
        labels_classification = labels_classification.to(device)
        labels_segmentation_infected = labels_segmentation_infected.to(device)
        labels_segmentation_lungs = labels_segmentation_lungs.to(device)

        # Zero the parameter gradient
        optimizer.zero_grad()

        # Forward pass
        outputs_classification, outputs_segmentation_lungs, outputs_segmentation_infected = model(inputs)
        
        outputs_classification = outputs_classification.type(torch.float32)
        outputs_segmentation_infected = outputs_segmentation_infected.type(torch.float32)
        outputs_segmentation_lungs = outputs_segmentation_lungs.type(torch.float32)

        labels_classification = labels_classification.type(torch.float32)
        labels_segmentation_infected = labels_segmentation_infected.type(torch.float32)
        labels_segmentation_lungs = labels_segmentation_lungs.type(torch.float32)
#         print(outputs_classification ,labels_classification)

        loss_classification = classification_loss_fn(outputs_classification, labels_classification)
        loss_segmentation_infected = segmentation_loss_fn(outputs_segmentation_infected, labels_segmentation_infected)
        loss_segmentation_lungs = segmentation_loss_fn(outputs_segmentation_lungs, labels_segmentation_lungs)
#         loss = (1/3 * loss_classification) + (1/3 * loss_segmentation_infected) + (1/3 * loss_segmentation_lungs)

        # loss = ((a * loss_classification) + (b * loss_segmentation_infected) + (c * loss_segmentation_lungs))/(10)
        loss =  (loss_segmentation_infected) + (a*( loss_classification) + ( loss_segmentation_lungs))



        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=loss.item())
        
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
         for batch_idx, (inputs, labels_classification,  labels_segmentation_lungs, labels_segmentation_infected) in enumerate(val_loader):
         # To device
            inputs = inputs.to(device)
            labels_classification = labels_classification.to(device)
            labels_segmentation_infected = labels_segmentation_infected.to(device)
            labels_segmentation_lungs = labels_segmentation_lungs.to(device)
            
            outputs_classification, outputs_segmentation_lungs, outputs_segmentation_infected = model(inputs)

            
            outputs_classification = outputs_classification.type(torch.float32)
            outputs_segmentation_infected = outputs_segmentation_infected.type(torch.float32)
            outputs_segmentation_lungs = outputs_segmentation_lungs.type(torch.float32)
            
            labels_classification = labels_classification.type(torch.float32)
            labels_segmentation_infected = labels_segmentation_infected.type(torch.float32)
            labels_segmentation_lungs = labels_segmentation_lungs.type(torch.float32)
            
#             print(outputs_classification ,labels_classification)
            
            loss_classification = classification_loss_fn(outputs_classification, labels_classification)
            loss_segmentation_infected = segmentation_loss_fn(outputs_segmentation_infected, labels_segmentation_infected)
            loss_segmentation_lungs = segmentation_loss_fn(outputs_segmentation_lungs, labels_segmentation_lungs)
    #         loss = (1/3 * loss_classification) + (1/3 * loss_segmentation_infected) + (1/3 * loss_segmentation_lungs)
            loss = (1/3 * loss_classification) + (1/3 * loss_segmentation_infected) + (1/3 * loss_segmentation_lungs)
            val_loss += loss.item() * inputs.size(0)

            outputs_classification = outputs_classification.argmax(1).detach().cpu().numpy()
            outputs_segmentation_infected = outputs_segmentation_infected.argmax(1)
            outputs_segmentation_lungs = outputs_segmentation_lungs.argmax(1)
            
            labels_classification = labels_classification.argmax(1).detach().cpu().numpy()
            labels_segmentation_infected = labels_segmentation_infected.argmax(1)
            labels_segmentation_lungs = labels_segmentation_lungs.argmax(1)
                
            
            pixel_acc_infected, dice_infected,iou_infected, precision_infected, recall_infected = calculate_overlap_metrics(labels_segmentation_infected, outputs_segmentation_infected,eps=1e-5)
            pixel_acc_lungs, dice_lungs,iou_lungs, precision_lungs, recall_lungs = calculate_overlap_metrics(labels_segmentation_lungs, outputs_segmentation_lungs,eps=1e-5)
            precision_classification = precision_score(labels_classification,outputs_classification,average='macro')
            recall_classification = recall_score(labels_classification,outputs_classification,average='macro')
            f1_score_classification = f1_score(labels_classification,outputs_classification,average='macro')
            
            pixel_acc_infected_meter.update(pixel_acc_infected,inputs.shape[0])
            dice_infected_meter.update(dice_infected,inputs.shape[0])
            iou_infected_meter.update(iou_infected,inputs.shape[0])
            precision_infected_meter.update(precision_infected,inputs.shape[0])
            recall_infected_meter.update(recall_infected,inputs.shape[0])

            pixel_acc_lungs_meter.update(pixel_acc_lungs,inputs.shape[0])
            dice_lungs_meter.update(dice_lungs,inputs.shape[0])
            iou_lungs_meter.update(iou_lungs,inputs.shape[0])
            precision_lungs_meter.update(precision_lungs,inputs.shape[0])
            recall_lungs_meter.update(recall_lungs,inputs.shape[0])

            precision_classification_meter.update(precision_classification,inputs.shape[0])
            recall_classification_meter.update(recall_classification,inputs.shape[0])
            f1_score_classification_meter.update(f1_score_classification,inputs.shape[0])
#             f1_score(y_true, y_pred, average='macro')
    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} \n \
    pixel_acc_infected: {pixel_acc_infected_meter.avg :.4f}, dice_infected: {dice_infected_meter.avg :.4f},iou_infected: {iou_infected_meter.avg :.4f}, precision_infected: {precision_infected_meter.avg :.4f}, recall_infected: {recall_infected_meter.avg :.4f} \n \
    pixel_acc_lungs: {pixel_acc_lungs_meter.avg :.4f}, dice_lungs: {dice_lungs_meter.avg :.4f},iou_lungs: {iou_lungs_meter.avg :.4f}, precision_lungs: {precision_lungs_meter.avg :.4f}, recall_lungs: {recall_lungs_meter.avg :.4f} \n\
     precision_classification: {precision_classification_meter.avg :.4f}, recall_classification: {recall_classification_meter.avg :.4f},f1_score_classification: {f1_score_classification_meter.avg :.4f} \n')

    # save models
    if f1_score_classification_meter.avg > best_acc:
        print(f"Best model found at epoch {epoch+1}, saving model")
        torch.save(model.state_dict(), "/kaggle/working/sample_best.ckpt") # only save best to prevent output memory exceed error
#         torch.save(model,'best.pth')
        best_acc = f1_score_classification_meter.avg
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break
    if epoch % save_every == 0 or epoch == num_epochs - 1:
        print(f"save model at epoch {epoch+1}, saving model")
        torch.save(model.state_dict(), f"/kaggle/working/epoch_{epoch}.ckpt")