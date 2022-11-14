import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
import cv2
import utils.model as model

def trainmodel(traindatas,batchsize):
    yolov1model = model.YOLOv1()
    batch=0
    batchimg=[]
    for img, target in traindatas:
        w = img.width
        h = img.height
        img = img.resize((448, 448), Image.BILINEAR)
        #a = ImageDraw.ImageDraw(img)
        for i in range(len(target['annotation']['object'])):
            target['annotation']['object'][i]['bndbox']['xmax'] = int(
                int(target['annotation']['object'][i]['bndbox']['xmax'])*448/w)
            target['annotation']['object'][i]['bndbox']['ymax'] = int(
                int(target['annotation']['object'][i]['bndbox']['ymax'])*448/h)
            target['annotation']['object'][i]['bndbox']['xmin'] = int(
                int(target['annotation']['object'][i]['bndbox']['xmin'])*448/w)
            target['annotation']['object'][i]['bndbox']['ymin'] = int(
                int(target['annotation']['object'][i]['bndbox']['ymin'])*448/h)
            #a.rectangle(((target['annotation']['object'][i]['bndbox']['xmin'], target['annotation']['object'][i]['bndbox']['ymin']), (
            #    target['annotation']['object'][i]['bndbox']['xmax'], target['annotation']['object'][i]['bndbox']['ymax'])), fill=None, outline='red', width=5)

        img = transforms.ToTensor()(img)
        img=img.unsqueeze(0)
        batchimg.append(img)
        if batch==batchsize:
            batch=0
            imgs=batchimg[0]
            del(batchimg[0])
            for img_ in batchimg[1:batchsize]:
                imgs=torch.cat([imgs,img_],dim=0)
            cudaimgs=imgs.cuda()
            re=yolov1model.forward(cudaimgs)
            print(re.shape)
            batchimg=[]

        else:
            batch+=1
        #print(i)
        #re=yolov1model.forward(img)
        #print(re.shape)
