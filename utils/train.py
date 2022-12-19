import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
import cv2
import utils.model as model

def trainmodel(traindatas,labeldatas,optimizer,batchsize,epoch):
    yolov1model = model.YOLOv1()
    lossfun= model.YOLOv1loss()
    for i in range(epoch):
        for b in range(len(traindatas)//batchsize):
            batchimgs=torch.cat(traindatas[b*batchsize:(b+1)*batchsize],dim=0)
            batchlabels = torch.cat(labeldatas[b*batchsize:(b+1)*batchsize],dim=0)
            preds=yolov1model.forward(batchimgs)
            loss=lossfun.loss(preds,batchlabels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

