import torchvision.datasets as datasets
import torchvision
import torch.nn as nn
import torch
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
import utils.VOC2007dataset as VOC2007
import utils.train as train

datas=VOC2007.VOC_Imagedataset()
m=train.trainmodel(datas.voc_testset,6)
"""
img,target=datas.voc_testset[6]
print(target['annotation']['object'][0]['bndbox']['xmax'])
w=img.width
h=img.height
img=img.resize((448,448), Image.BILINEAR)
a = ImageDraw.ImageDraw(img)

for i in range(len(target['annotation']['object'])):
    target['annotation']['object'][i]['bndbox']['xmax']=int(int(target['annotation']['object'][i]['bndbox']['xmax'])*448/w)
    target['annotation']['object'][i]['bndbox']['ymax']=int(int(target['annotation']['object'][i]['bndbox']['ymax'])*448/h)
    target['annotation']['object'][i]['bndbox']['xmin']=int(int(target['annotation']['object'][i]['bndbox']['xmin'])*448/w)
    target['annotation']['object'][i]['bndbox']['ymin']=int(int(target['annotation']['object'][i]['bndbox']['ymin'])*448/h)
    a = ImageDraw.ImageDraw(img)
    a.rectangle(((target['annotation']['object'][i]['bndbox']['xmin'], target['annotation']['object'][i]['bndbox']['ymin']),(target['annotation']['object'][i]['bndbox']['xmax'], target['annotation']['object'][i]['bndbox']['ymax'])), fill=None, outline='red', width=5)

            
    
img.show()
"""