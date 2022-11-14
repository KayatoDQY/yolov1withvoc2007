import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__
        self.layers = self.create_layer()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            #print(x.shape)
        return x

    def create_layer(self):
        layer = []
        layer.append(nn.Conv2d(in_channels=3, out_channels=64,
                     kernel_size=7, stride=2, padding=3).cuda())
        layer.append(nn.BatchNorm2d(64).cuda())
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2).cuda())

        layer.append(nn.Conv2d(in_channels=64, out_channels=192,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(192).cuda())
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2).cuda())

        layer.append(nn.Conv2d(in_channels=192, out_channels=128,
                     kernel_size=1, stride=1, padding=0).cuda())
        layer.append(nn.BatchNorm2d(128).cuda())
        layer.append(nn.Conv2d(in_channels=128, out_channels=256,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(256).cuda())
        layer.append(nn.Conv2d(in_channels=256, out_channels=256,
                     kernel_size=1, stride=1, padding=0).cuda())
        layer.append(nn.BatchNorm2d(256).cuda())
        layer.append(nn.Conv2d(in_channels=256, out_channels=512,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(512).cuda())
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2).cuda())

        layer.append(nn.Conv2d(in_channels=512, out_channels=256,
                     kernel_size=1, stride=1, padding=0).cuda())
        layer.append(nn.BatchNorm2d(256).cuda())
        layer.append(nn.Conv2d(in_channels=256, out_channels=512,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(512).cuda())
        layer.append(nn.Conv2d(in_channels=512, out_channels=256,
                     kernel_size=1, stride=1, padding=0).cuda())
        layer.append(nn.BatchNorm2d(256).cuda())
        layer.append(nn.Conv2d(in_channels=256, out_channels=512,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(512).cuda())
        layer.append(nn.Conv2d(in_channels=512, out_channels=256,
                     kernel_size=1, stride=1, padding=0).cuda())
        layer.append(nn.BatchNorm2d(256).cuda())
        layer.append(nn.Conv2d(in_channels=256, out_channels=512,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(512).cuda())
        layer.append(nn.Conv2d(in_channels=512, out_channels=256,
                     kernel_size=1, stride=1, padding=0).cuda())
        layer.append(nn.BatchNorm2d(256).cuda())
        layer.append(nn.Conv2d(in_channels=256, out_channels=512,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(512).cuda())
        layer.append(nn.Conv2d(in_channels=512, out_channels=512,
                     kernel_size=1, stride=1, padding=0).cuda())
        layer.append(nn.BatchNorm2d(512).cuda())
        layer.append(nn.Conv2d(in_channels=512, out_channels=1024,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(1024).cuda())
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2).cuda())

        layer.append(nn.Conv2d(in_channels=1024, out_channels=512,
                     kernel_size=1, stride=1, padding=0).cuda())
        layer.append(nn.BatchNorm2d(512).cuda())
        layer.append(nn.Conv2d(in_channels=512, out_channels=1024,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(1024).cuda())
        layer.append(nn.Conv2d(in_channels=1024, out_channels=512,
                     kernel_size=1, stride=1, padding=0).cuda())
        layer.append(nn.BatchNorm2d(512).cuda())
        layer.append(nn.Conv2d(in_channels=512, out_channels=1024,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(1024).cuda())
        layer.append(nn.Conv2d(in_channels=1024, out_channels=1024,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(1024).cuda())
        layer.append(nn.Conv2d(in_channels=1024, out_channels=1024,
                     kernel_size=3, stride=2, padding=1).cuda())
        layer.append(nn.BatchNorm2d(1024).cuda())

        layer.append(nn.Conv2d(in_channels=1024, out_channels=1024,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(1024).cuda())
        layer.append(nn.Conv2d(in_channels=1024, out_channels=1024,
                     kernel_size=3, stride=1, padding=1).cuda())
        layer.append(nn.BatchNorm2d(1024).cuda())

        layer.append(nn.Flatten().cuda())
        layer.append(nn.Linear(in_features=50176, out_features=4096).cuda())
        layer.append(nn.LeakyReLU(0.1).cuda())
        layer.append(nn.Linear(in_features=4096, out_features=1470).cuda())
        return layer


# 测试网络连通
"""
x=torch.rand(1,3,448,448)
model=YOLOv1()
x=model.forward(x)
print(x)
"""


class YOLOv1loss(nn.Module):
    def __init__(self):
        super(YOLOv1loss, self).__init__
        self.coord = 5
        self.nooobj = 0.5

    def calculateIOU(self, box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:,
                                                  0], box1[:, 1], box1[:, 2], box1[:, 3]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:,
                                                  0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)

        inter_s = torch.clamp(inter_x2-inter_x1+1, min=0) * \
            torch.clamp(inter_y2-inter_y1+1, min=0)

        box1_s = (box1_x2-box1_x1+1)*(box1_y2-box1_y1+1)
        box2_s = (box2_x2-box2_x1+1)*(box2_y2-box2_y1+1)

        iou = inter_s/(box1_s+box2_s-inter_s)

        return iou

    def loss(self, pre, lab):
        # pre有7*7*30组，
        # lab有7*7*（框的大小位置4,20类的概率）
        eachlabsize = 4+20
        eachpresize = 5+5+20
        addressloss = 0
        sizeloss = 0
        objloss=0
        noobjloss=0
        for i in range(7*7):
            eachpre = pre[:, i*eachpresize:(i+1)*eachpresize-1]
            eachlab = lab[:, i*eachlabsize:(i+1)*eachlabsize-1]
            iou1 = self.calculateIOU(eachpre[:, 0:3], eachlab[:, 0:3])
            iou2 = self.calculateIOU(eachpre[:, 5:8], eachlab[:, 0:3])

            if iou1 >= iou2:
                addressloss = addressloss+self.coord * \
                    ((eachpre[:, 0]-eachlab[:, 0]) **
                     2+(eachpre[:, 1]-eachlab[:1])**2)
                sizeloss = sizeloss+self.coord*(((eachpre[:, 2]-eachpre[:, 0])**0.5-(eachlab[:, 2]-eachlab[:, 0])**0.5)**2+(
                    (eachpre[:, 3]-eachpre[:, 1])**0.5-(eachlab[:, 3]-eachlab[:, 1])**0.5)**2)
                objloss = objloss+(eachpre[:,4]-iou1)**2
                noobjloss=noobjloss+self.nooobj*(eachpre[:,9]-iou2)**2

            elif iou1 < iou2:
                addressloss = addressloss+self.coord * \
                    ((eachpre[:, 5]-eachlab[:, 0]) **
                     2+(eachpre[:, 6]-eachlab[:1])**2)
                sizeloss = sizeloss+self.coord*(((eachpre[:, 7]-eachpre[:, 5])**0.5-(eachlab[:, 2]-eachlab[:, 0])**0.5)**2+(
                    (eachpre[:, 8]-eachpre[:, 6])**0.5-(eachlab[:, 3]-eachlab[:, 1])**0.5)**2)
                objloss = objloss+(eachpre[:,9]-iou2)**2
                noobjloss=noobjloss+self.nooobj*(eachpre[:,4]-iou1)**2
                
            classloss=classloss+torch.sum((eachpre[10:29]-eachlab[4,23])**2)

        return addressloss+sizeloss+objloss+noobjloss+classloss
        