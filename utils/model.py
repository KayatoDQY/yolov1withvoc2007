import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__
        self.layers=self.create_layer()


    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
            print(x.shape)
        return x
    

    def create_layer(self):
        layer=[]
        layer.append(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3))
        layer.append(nn.BatchNorm2d(64))
        layer.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layer.append(nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(192))
        layer.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layer.append(nn.Conv2d(in_channels=192,out_channels=128,kernel_size=1,stride=1,padding=0))
        layer.append(nn.BatchNorm2d(128))
        layer.append(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(256))
        layer.append(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1,padding=0))
        layer.append(nn.BatchNorm2d(256))
        layer.append(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(512))
        layer.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layer.append(nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0))
        layer.append(nn.BatchNorm2d(256))
        layer.append(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(512))
        layer.append(nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0))
        layer.append(nn.BatchNorm2d(256))
        layer.append(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(512))
        layer.append(nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0))
        layer.append(nn.BatchNorm2d(256))
        layer.append(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(512))
        layer.append(nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0))
        layer.append(nn.BatchNorm2d(256))
        layer.append(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(512))
        layer.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,stride=1,padding=0))
        layer.append(nn.BatchNorm2d(512))
        layer.append(nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(1024))
        layer.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layer.append(nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1,padding=0))
        layer.append(nn.BatchNorm2d(512))
        layer.append(nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(1024))
        layer.append(nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1,padding=0))
        layer.append(nn.BatchNorm2d(512))
        layer.append(nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(1024))
        layer.append(nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(1024))
        layer.append(nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=2,padding=1))
        layer.append(nn.BatchNorm2d(1024))

        layer.append(nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(1024))
        layer.append(nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1))
        layer.append(nn.BatchNorm2d(1024))

        layer.append(nn.Flatten())
        layer.append(nn.Linear(in_features=50176,out_features=4096))
        layer.append(nn.LeakyReLU(0.1))
        layer.append(nn.Linear(in_features=4096,out_features=1470))
        return layer

#测试网络连通
"""
x=torch.rand(1,3,448,448)
model=YOLOv1()
x=model.forward(x)
print(x)
"""


