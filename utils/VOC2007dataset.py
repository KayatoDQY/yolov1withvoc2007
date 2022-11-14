import torchvision.datasets as datasets
import torchvision
import torch
import numpy as np
import cv2


class VOC_Imagedataset():
    def __init__(self,datasetfilename='dataset'):
        self.voc_trainset = datasets.VOCDetection(datasetfilename, year='2007', image_set='train', download=False)
        self.voc_valset = datasets.VOCDetection(datasetfilename, year='2007', image_set='val', download=False)
        self.voc_testset = datasets.VOCDetection(datasetfilename, year='2007', image_set='test', download=False)
        print("Number of train",len(self.voc_trainset))
        print("Number of val",len(self.voc_valset))
        print("Number of test",len(self.voc_testset))


