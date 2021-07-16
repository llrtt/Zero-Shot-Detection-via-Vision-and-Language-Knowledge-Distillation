from matplotlib import image
from torch import nn
from torch.nn.modules.conv import Conv2d
from torch.utils.data.dataset import T
from torchvision.models.detection import backbone_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import dataloader
import numpy as np
import vocDataset
import clip
import torch
import Modified
import faster_rcnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


class VILD(nn.Module):
    def __init__(self):
        super(VILD, self).__init__()

        self.backbone = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
        self.backbone.box_nms_thresh = 0.9
        self.backbone.roi_heads.box_predictor = Modified.FastRCNNPredictor(
            1024, 2)
        self.backbone.load_state_dict(torch.load("maskrcnn5.pt"))

        box_head = Modified.TwoMLPHead(
            self.backbone.backbone.out_channels *
            self.backbone.roi_heads.box_roi_pool.output_size[0] ** 2,
            1024)

        box_head.fc6 = self.backbone.roi_heads.box_head.fc6
        box_head.fc7 = self.backbone.roi_heads.box_head.fc7

        self.backbone.roi_heads.box_head = box_head

        self.region = backbone_utils.resnet.resnet18()
        self.region.conv1 = nn.Conv2d(256, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.region.fc = nn.Linear(512, 512)

        self.model, self.preprocess = clip.load('ViT-B/32', device, jit=False)

    def VILD_image(self, proposals, images):
        # 获得变换之后的图片和原图的尺寸比例
        image_sizes = torch.tensor(
            self.backbone.transform(images)[0].image_sizes)
        x_ratio = torch.tensor([image.shape[1] for image in images]) / \
            torch.tensor([image_size[0] for image_size in image_sizes])
        y_ratio = torch.tensor([image.shape[2] for image in images]) / \
            torch.tensor([image_size[1] for image_size in image_sizes])
        x_ratio = x_ratio.unsqueeze(1).repeat([1, proposals.shape[1]])
        y_ratio = y_ratio.unsqueeze(1).repeat([1, proposals.shape[1]])

        # 将proposal变换为原来的图片对应的尺寸及1.5倍原来尺寸
        proposals[:, :, 0] = proposals[:, :, 0] * x_ratio
        proposals[:, :, 1] = proposals[:, :, 1] * y_ratio
        proposals[:, :, 2] = proposals[:, :, 2] * x_ratio
        proposals[:, :, 3] = proposals[:, :, 3] * y_ratio

        # proposals1_5 = proposals*1.5

        for image, i in zip(images, range(proposals.shape[0])):
            proposals[i][:, 0] = proposals[i][:, 0].clamp(
                0, image.shape[2]-2)
            proposals[i][:, 1] = proposals[i][:, 1].clamp(
                0, image.shape[1]-2)
            proposals[i][:, 2] = proposals[i][:, 2].clamp(
                0, image.shape[2]-2)
            proposals[i][:, 3] = proposals[i][:, 3].clamp(
                0, image.shape[1]-2)
            print(proposals[i])
            print(image.shape)
        width = proposals[:, :, 3] - proposals[:, :, 1]
        height = proposals[:, :, 2] - proposals[:, :, 0]
        print(width)

        # 将小的矩形框进行调整
        for i in range(len(images)):
            for j in range(proposals.shape[1]):
                if(width[i, j] <= 0):
                    if(proposals[i, j, 3] <= 2):
                        proposals[i, j, 3] = proposals[i, j, 3]+2
                        proposals[i, j, 1] = proposals[i, j, 3]+1
                    else:
                        proposals[i, j, 3] = proposals[i, j, 3]-1
                        proposals[i, j, 1] = proposals[i, j, 3]-2
                if(height[i, j] <= 0):
                    if(proposals[i, j, 2] <= 2):
                        proposals[i, j, 2] = proposals[i, j, 2]+2
                        proposals[i, j, 0] = proposals[i, j, 2]+1
                    else:
                        proposals[i, j, 2] = proposals[i, j, 2]-1
                        proposals[i, j, 0] = proposals[i, j, 2]-2

        # 将变换后的proposals输入clip获得1000x512的image_embedding

        postprocess = transforms.ToPILImage()
        proposals = proposals.long()
        image_embeddings = torch.empty(
            len(images), proposals.shape[1], 512)

        for image, j in zip(images, range(len(images))):
            for i in range(proposals.shape[1]):
                image_input = self.preprocess(postprocess(
                    image[:, proposals[j, i, 1]:proposals[j, i, 3], proposals[j, i, 0]:proposals[j, i, 2]])).unsqueeze(0).to(device)
                image_embedding = self.model.encode_image(image_input)
                image_embeddings[j, i] = image_embedding

        # 归一化
        v = torch.linalg.norm(image_embeddings, ord=1,
                              dim=2).unsqueeze(2).repeat([1, 1, 512])
        image_embeddings = image_embeddings / v

        return image_embeddings
    
    def VILD_text(self):
        if self.training:
        # 生成text_embedding，训练时只用到VOC数据集中前十个类
            text_inputs = torch.cat(
                    [clip.tokenize(f"a photo of a {c}") for c in vocDataset.classes[:10]]).to(device)
        else:
            text_inputs = torch.cat(
                    [clip.tokenize(f"a photo of a {c}") for c in vocDataset.classes[:]]).to(device)
        text_embedding = self.model.encode_text(text_inputs)
        return text_embedding

    def forward(self, images):
        # backbone已经训练好
        self.backbone.eval()

        # backbone输入应该为image数组
        detection = self.backbone(images)

        region_embedding = self.region(
            self.backbone.roi_heads.box_head.features).reshape(2, int(len(self.backbone.roi_heads.box_head.features)/len(images)), 512)

        if self.training:
            proposals = torch.stack(
                [proposal for proposal in self.backbone.proposals])
            images_embedding = self.VILD_image(proposals, images)
            text_embedding = self.VILD_text()

            return proposals, region_embedding

        return region_embedding


if __name__ == '__main__':
    preprocess = vocDataset.get_transform(False)  # 主要作用是将图片数据转成tensor传入显存
    postprocess = transforms.ToPILImage()

    dataset = vocDataset.vocData(
        "/home/llrt/文档/VOCdevkit/VOC2012", transform=preprocess)
    model0 = VILD()
    model0.eval()

    image0, target = dataset[0]
    image1, target = dataset[2]

    img = []
    img.append(image0)
    img.append(image1)
    model0(img)

    # model0.backbone.roi_heads.box_predictor.box
    print('wanle')
    # image = VILD_image()
    # print(image.backbone.proposals)
