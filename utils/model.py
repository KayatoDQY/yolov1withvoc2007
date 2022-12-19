import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__
        self.layers = self.create_layer()

    def forward(self, x):
        batchsize = x.shape[0]
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        x = x.view(batchsize, 30, 7, 7)
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
        pre = pre.double()
        lab = lab.double()
        batchsize = pre.shape[0]
        addressloss = 0
        sizeloss = 0
        objloss = 0
        noobjloss = 0
        classloss = 0
        for i in range(batchsize):
            eachpre = pre[i, :, :, :]
            eachlab = lab[i, :, :, :]
            for px in range(7):
                for py in range(7):
                    if eachlab[4, px, py] == 1:
                        iou1 = self.calculateIOU(
                            eachpre[0:4, px, py], eachlab[0:4, px, py])
                        iou2 = self.calculateIOU(
                            eachpre[5:9, px, py], eachlab[5:9, px, py])
                        if iou1 >= iou2:
                            cxp = px*64+eachpre[0, px, py]
                            cyp = py*64+eachpre[1, px, py]
                            cxl = px*64+eachlab[0, px, py]
                            cyl = py*64+eachlab[1, px, py]
                            addressloss = addressloss+self.coord * \
                                ((cxl-cxp)**2+(cyl-cyp)**2)
                            hp = eachpre[2, px, py]
                            wp = eachpre[3, px, py]
                            hl = eachlab[2, px, py]
                            wl = eachlab[3, px, py]
                            sizeloss = sizeloss+self.coord * \
                                ((hp**0.5-hl**0.5)**2+(wp**0.5-wl**0.5)**2)
                            cp = eachpre[9, px, py]
                            cl = iou1
                            noobjloss = noobjloss+self.nooobj*(cp-cl)**2
                        elif iou1 < iou2:
                            cxp = px*64+eachpre[5, px, py]
                            cyp = py*64+eachpre[6, px, py]
                            cxl = px*64+eachlab[5, px, py]
                            cyl = py*64+eachlab[6, px, py]
                            addressloss = addressloss+self.coord * \
                                ((cxl-cxp)**2+(cyl-cyp)**2)
                            hp = eachpre[7, px, py]
                            wp = eachpre[8, px, py]
                            hl = eachlab[7, px, py]
                            wl = eachlab[8, px, py]
                            sizeloss = sizeloss+self.coord * \
                                ((hp**0.5-hl**0.5)**2+(wp**0.5-wl**0.5)**2)
                            cp = eachpre[9, px, py]
                            cl = iou2
                            noobjloss = noobjloss+self.nooobj*(cp-cl)**2
                    else:
                        noobjloss = noobjloss+self.nooobj * \
                            (eachpre[4, px, py]+eachpre[9, px, py])**2
                    classloss = classloss + \
                        torch.sum(
                            (self.pred[10:, px, py] - eachlab[10:, px, py]) ** 2)

        return (addressloss+sizeloss+objloss+noobjloss+classloss)/batchsize
