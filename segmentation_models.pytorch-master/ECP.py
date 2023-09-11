import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import torch
import numpy as np
import imgaug.augmenters as iaa
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
    
        for _class in class_dir:
            self.class_path = os.path.join(self.rootpth,self.mode.capitalize())
            self.img_path = os.path.join(self.class_path,_class,'images')
            img_lst = os.listdir(self.img_path)
            img_lst.sort()
            for img in img_lst:
                self.imgs.append((img,self.class_cvt[_class]))
        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_color = iaa.Sequential([
            iaa.LinearContrast((0.4, 1.6)),
            ])
        self.trans_train = iaa.Sequential([
            iaa.Fliplr(0.5),
            ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        self.img_path = os.path.join(self.class_path,list(self.class_cvt.keys())[impth[1]],'images')
                
        img =  cv2.imread(os.path.join(self.img_path,impth[0]))
        img = cv2.resize(img,self.cropsize, cv2.INTER_LINEAR)
        label = F.one_hot(torch.tensor(impth[1]), num_classes=3)
        
        mask_path = self.img_path.replace('images','lung masks')
        lung_mask = cv2.imread(os.path.join(mask_path,impth[0]),0)
        lung_mask = cv2.resize(lung_mask,self.cropsize, cv2.INTER_NEAREST ).astype(np.int64)        
        
        infect_path = self.img_path.replace('images','infection masks')
        infect_mask = cv2.imread(os.path.join(infect_path,impth[0]),0)
        infect_mask = cv2.resize(infect_mask,self.cropsize, cv2.INTER_NEAREST ).astype(np.int64)
      
        if self.mode == 'train':
            det_tf = self.trans_train.to_deterministic()
            img = det_tf.augment_image(img)
            lung_mask = det_tf.augment_image( lung_mask)
            infect_mask = det_tf.augment_image(infect_mask)

        if self.mode != 'test':
            img = self.to_tensor(img)        
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
    
    
class ConvBnRelu(nn.Module):
    """Convenience layer combining a Conv2d, BatchNorm2d, and a ReLU activation.

    Original source of this code comes from
    https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 norm_layer=nn.BatchNorm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvRelu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(inplace=True))

class MobileNetV3_Modified(nn.Module):
    def __init__(self):
        super(MobileNetV3_Modified, self).__init__()
        mobile_net_v3_small = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        
        # Extract only features layer
        self.blocks = nn.ModuleList()
        self.num_layers = len(mobile_net_v3_small.features) 
        for i in range(self.num_layers):
            if (i == 4):
                mobile_net_v3_small.features[i].block[1][0].stride = (1, 1)
            self.blocks.append(mobile_net_v3_small.features[i])
            
    def forward(self, x):
        outputs = []
        for i in range(self.num_layers):
            x = self.blocks[i](x)
            if (i == 0 or i == 8 or i == self.num_layers-1):
                outputs.append(x)
        return outputs

class SegHead(nn.Module):
    def __init__(self, d_model=128, n_classes=2):
        super(SegHead, self).__init__()
        self.reduce_conv = ConvRelu(576, d_model, 1)
        self.up_conv1 = ConvRelu(48 + d_model, d_model, 1)
        self.up_conv2 = nn.Conv2d(16 + d_model, n_classes, 1)
            
    def forward(self, x1, x2, x3, x_shape):        
        x3 = self.reduce_conv(x3)
        x3 = F.interpolate(x3, x2.shape[2:], mode='bilinear', align_corners=True)
                
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.up_conv1(x2)
        x2 = F.interpolate(x2, x1.shape[2:], mode='bilinear', align_corners=True)
        
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.up_conv2(x1)
        output = F.interpolate(x1, x_shape[2:], mode='bilinear', align_corners=True)        
        return output
    
class Multitask_MobileV3Smal_LRASPP(nn.Module):
    def __init__(self, backbone,lung_head,infected_head, num_class_classify=3):
        super(Multitask_MobileV3Smal_LRASPP, self).__init__()
        self.backbone = backbone()
        self.lung_head = lung_head()
        self.infected_head = infected_head()
        self.classify_branch =  nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Linear(in_features=1024, out_features=num_class_classify, bias=True))
        
    def forward(self, x):
        x_shape = x.shape
        x1, x2, x3 = self.backbone(x)
        lung_output= self.lung_head(x1, x2, x3, x_shape)
        infect_output  = self.infected_head(x1, x2, x3, x_shape)
        y = self.classify_branch(x3)
        return y, lung_output, infect_output        
    


