import transforms as T
from posix import listdir
from torchvision.io import image, read_image
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
import lxml
import xml.dom.minidom as dom
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def collate_fn(batch):
    return tuple(zip(*batch))

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def bbox_to_rect(bbox, color):
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class vocData(Dataset):
    def __init__(self, train_class=20, data_path="/home/llrt/文档/VOCdevkit/VOC2012", transform=None, target_transform=None):
        self.SegmentationClass = os.path.join(data_path, 'SegmentationClass')
        self.Annotations = os.path.join(data_path, 'Annotations')
        self.ImageSets = os.path.join(data_path, 'ImageSets')
        self.JPEGImages = os.path.join(data_path, 'JPEGImages')
        self.labels = os.path.join(data_path, 'labels')
        self.train_classes = int(train_class)

        self.imagePath = []
        self.xmlPath = []
        self.hasObject = []

        self.transform = transform
        self.target_transform = target_transform

        file = open(self.ImageSets+'/Main/bicycle_train.txt', 'r+')
        lines = [each for each in file.readlines()]
        self.imagePath = [self.JPEGImages + '/' + each.split(
            " ")[0]+'.jpg' for each in lines]
        self.hasObject = [each.split(" ")[1].strip() for each in lines]
        self.xmlPath = [self.Annotations + '/' + each.split(
            " ")[0]+'.xml' for each in lines]
        # 筛选没有目标的图片
        # imagePath = []
        # xmlPath = []
        # for has, image, xml in zip(self.hasObject, self.imagePath, self.xmlPath):
        #     if has == '-1':
        #         continue
        #     else:
        #         imagePath.append(image)
        #         xmlPath.append(xml)
        # self.imagePath = imagePath
        # self.xmlPath = xmlPath
        file.close()

    def __len__(self):
        return len(self.imagePath)

    def __getitem__(self, index):
        image = Image.open(self.imagePath[index]).convert("RGB")
        DOMTree = dom.parse(self.xmlPath[index])
        collection = DOMTree.documentElement
        Objects = collection.getElementsByTagName("object")
        boxes = []
        labels = []
        names = {}
        for i, elements in enumerate(classes):
            names.update({elements: i})
        for Object in Objects:
            bndboxes = Object.getElementsByTagName("bndbox")
            # for bndbox in bndboxes:
            xmin = int(bndboxes[0].getElementsByTagName(
                "xmin")[0].childNodes[0].data)
            ymin = int(bndboxes[0].getElementsByTagName(
                "ymin")[0].childNodes[0].data)
            xmax = int(bndboxes[0].getElementsByTagName(
                "xmax")[0].childNodes[0].data)
            ymax = int(bndboxes[0].getElementsByTagName(
                "ymax")[0].childNodes[0].data)
            if names[Object.getElementsByTagName("name")[0].childNodes[0].data] >= self.train_classes:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(names[Object.getElementsByTagName(
                "name")[0].childNodes[0].data])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # 种类不敏感模型就取消注释
        # labels = torch.ones((len(self.imagePath,)), dtype=torch.int64)
        image_id = torch.tensor([index])
        if(int(len(boxes)) == 0):
            area = 0
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(Objects),), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target


if __name__ == '__main__':
    preprocess = get_transform(False)
    data = vocData(data_path="/home/llrt/文档/VOCdevkit/VOC2012", transform=preprocess)
    loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)
    print(len(data))

    for imgs, targets in loader:
        images = list([img for img in imgs])
        target = [{k: v for k, v in target.items()}
                  for target in targets]
        plt.cla()
        postprocess = transforms.ToPILImage()
        fig = plt.imshow(postprocess(images[0]))
        print(targets[0]["boxes"])
        if(int(len(targets[0]["boxes"])) != 0):
            box = target[0]['boxes'][0].long()
            for bbox in target[0]['boxes']:
                fig.axes.add_patch(bbox_to_rect(bbox, 'blue'))
        plt.pause(1)
