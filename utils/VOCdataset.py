import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np
from PIL import Image


class VOCdataset():
    # 1.读取数据集
    # 2.将图像设置为1,3,448,448大小tensor
    # 3.将标签转换为1,30,7,7大小的tensor,gxc,gyc,w,h,c,p[20]
    def __init__(self, datasetfilename: str = 'dataset', mode: str = 'train'):
        self.CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
        self.rdatas = datasets.VOCDetection(
            datasetfilename, year='2007', image_set=mode, download=False)
        self.imgs=[]
        self.labers=[]
        for img, label in self.rdatas:
            rw, rh, tensorimg = self.pil2tensor480(img)
            tensorlaber = self.laber2tensor(
                rw, rh, label['annotation']['object'])
            self.imgs.append(tensorimg)
            self.labers.append(tensorlaber)

    def pil2tensor480(self, img):
        w = img.width
        h = img.height
        img = img.resize((448, 448), Image.BILINEAR)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        return w, h, img

    def laber2tensor(self, rw, rh, bboxs):
        nplabel = np.zeros((30, 7, 7))
        for i in range(len(bboxs)):
            cclass = self.CLASSES.index(bboxs[i]['name'])
            bbox = [int(int(bboxs[i]['bndbox']['xmin'])*480/rw),
                    int(int(bboxs[i]['bndbox']['ymin'])*480/rh),
                    int(int(bboxs[i]['bndbox']['xmax'])*480/rw),
                    int(int(bboxs[i]['bndbox']['ymax'])*480/rh), ]
            ggrid, bbox = self.xyxy2pxcpycwh(bbox)
            nplabel[0:5,ggrid[0],ggrid[1]]=[bbox[0],bbox[1],bbox[2],bbox[3],1]
            nplabel[5:10,ggrid[0],ggrid[1]]=[bbox[0],bbox[1],bbox[2],bbox[3],1]
            nplabel[10+cclass,ggrid[0],ggrid[1]]=1
        #nplabel=nplabel.reshape(1,-1)
        tensorlaber=torch.from_numpy(nplabel).unsqueeze(0)
]\
    
    
            print(tensorlaber.shape)
        return tensorlaber

    def xyxy2pxcpycwh(self, bbox):
        xc = (bbox[0]+bbox[2])//2
        yc = (bbox[1]+bbox[3])//2
        px = xc//64-1
        py = yc//64-1
        pxc = xc % 64
        pyc = yc % 64
        bw = bbox[2]-bbox[0]
        bh = bbox[3]-bbox[2]
        return [px, py], [pxc, pyc, bw, bh]

data=VOCdataset()