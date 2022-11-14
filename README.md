# yolov1withvoc2007

## VOC2007数据集下载

[Pascal VOC Dataset Mirror (pjreddie.com)](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

![VOCweb](https://github.com/KayatoDQY/yolov1withvoc2007/blob/main/imgofreadme/VOCweb.png)

这里使用VOC2007，需要下载` Train/Validation Data `和`Test Data With Annotations`

下载后解压两个文件夹均为以下格式

![VOCfile](https://github.com/KayatoDQY/yolov1withvoc2007/blob/main/imgofreadme/VOCfile.png)

拖拽这两个文件夹直接合并

在`Annotations`中包含了所有图片的xml标注文件

在`JPEGImages`中包括了所有jpg格式图片

`SegmentationClass`和`SegmentationObject`中为语义分割的标签，在此不会使用

重要的在__ImageSets__中,其中`Layout`和`Segmentation`为Pascal VOC比赛的其他两个子任务，主要任务在_Main_中。总共有20类

```
person

bird cat cow dog horse sheep

aeroplane bicycle boat bus car motorbike train

bottle chair dining table pottedplant sofa tv/monitor
```

如aeroplane含有一下4个文件

```
aeroplane_train.txt
aeroplane_trainval.txt
aeroplane_val.txt
aeroplane_test.txt
```

以及总体的train.txt、trainval.txt、val.txt、test.txt的4个文件，总共84个文件

*注意：train_train.txt等前者是火车的意思，后者是训练的意思*



## 读取数据集

采用pytorch自带函数读取，将数据集按照上述配置分别读取

```python
import torchvision.datasets as datasets
voc_trainset = datasets.VOCDetection(datasetfilename, year='2007', image_set='train', download=False)
voc_valset = datasets.VOCDetection(datasetfilename, year='2007', image_set='val', download=False)
voc_testset = datasets.VOCDetection(datasetfilename, year='2007', image_set='test', download=False)
```

