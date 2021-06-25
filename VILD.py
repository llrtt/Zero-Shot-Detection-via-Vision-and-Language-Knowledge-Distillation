from torch.utils.data import dataloader
from torchvision.models import detection
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from MaskRCNN import maskRCNN
import torch
import clip
import torchvision
import loadVOC
from PIL import Image
import vocDataset
import torchvision.transforms as transforms
import utils
from torchvision import models
import matplotlib.pyplot as plt
import torchvision.models.detection.faster_rcnn
import nms
import numpy as np


def get_transform(train):
    transform = []
    transform.append(transforms.ToTensor())
    return transforms.Compose(transform)


def VILD_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = vocDataset.get_transform(True)  # 主要作用是将图片数据转成tensor传入显存
    dataset = vocDataset.vocData(
        "/home/llrt/文档/VOCdevkit/VOC2012", transform=preprocess)
    print(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn, pin_memory=True)

    num_epochs = 100
    model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
    model.box_nms_thresh = 0.9
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    # model.load_state_dict(torch.load("maskrcnn5.pt"))
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.0007, momentum=0.8, weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.9)
    print(model)
    model.train()
    postprocess = transforms.ToPILImage()
    for i in range(num_epochs):
        model.train()
        for (imgs, targets), j in zip(loader, range(len(dataset))):
            images = list([img.to(device) for img in imgs])
            target = [{k: v.to(device) for k, v in target.items()}
                      for target in targets]
            loss_dict = model(images, target)
            losses = sum(loss for loss in loss_dict.values())
            print("epoch:{i} iteration:{j} loss:{losses}".format(
                i=i, j=j, losses=losses))
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            fig = plt.imshow(postprocess(images[0].cpu()))
            plt.pause(1)
        if lr_scheduler is not None:
            lr_scheduler.step()

        model.eval()
        images0, targets0 = dataset[i]
        img = []
        img.append(images0.to(device))
        targets0 = model(img)
        print(targets0)
        postprocess = transforms.ToPILImage()
        plt.cla()
        fig = plt.imshow(postprocess(images0.cpu()))
        for bbox, score in zip(targets0[0]['boxes'], targets0[0]['scores']):
            if(score < 0.3):
                continue
            fig.axes.add_patch(vocDataset.bbox_to_rect(
                bbox.cpu().detach(), 'blue'))
            # fig.axes.add_patch(vocDataset.bbox_to_rect(
            #     target['boxes'][1].cpu().detach(), 'blue'))
        plt.pause(1)
        torch.save(model.state_dict(), 'maskrcnn5.pt')


def VILD_evluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
    model.box_nms_thresh = 0.9
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    model.load_state_dict(torch.load("maskrcnn5.pt"))
    model.to(device)
    model.eval()

    preprocess = vocDataset.get_transform(False)  # 主要作用是将图片数据转成tensor传入显存
    postprocess = transforms.ToPILImage()

    dataset = vocDataset.vocData(
        "/home/llrt/文档/VOCdevkit/VOC2012", transform=preprocess)
    for i in range(100):
        image, target = dataset[i]
        img = []
        img.append(image.to(device))

        detections = model(img)

        fig = plt.imshow(postprocess(image.cpu()))

        bboxes, scores = nms.nms(detections[0]['boxes'].cpu().detach(
        ).numpy(), detections[0]['scores'].cpu().detach().numpy(), 0.2)

        for bbox, score in zip(bboxes, scores):
            if score < 0.2:
                continue
            fig.axes.add_patch(vocDataset.bbox_to_rect(
                bbox, 'blue'))
        plt.show()


if __name__ == '__main__':
    VILD_train()
