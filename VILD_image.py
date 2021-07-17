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
import nms

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
        self.region.to(device)

        self.model, self.preprocess = clip.load('ViT-B/32', device, jit=False)

    def VILD_image(self, images):
        # 获得变换之后的图片和原图的尺寸比例
        image_sizes = torch.tensor(
            self.backbone.transform(images)[0].image_sizes)
        x_ratio = torch.tensor([image.shape[1] for image in images]) / \
            torch.tensor([image_size[0] for image_size in image_sizes])
        y_ratio = torch.tensor([image.shape[2] for image in images]) / \
            torch.tensor([image_size[1] for image_size in image_sizes])
        x_ratio = x_ratio.unsqueeze(1).repeat(
            [1, self.proposals.shape[1]]).to(device)
        y_ratio = y_ratio.unsqueeze(1).repeat(
            [1, self.proposals.shape[1]]).to(device)

        # 将proposal变换为原来的图片对应的尺寸及1.5倍原来尺寸
        self.proposals[:, :, 0] = self.proposals[:, :, 0] * x_ratio
        self.proposals[:, :, 1] = self.proposals[:, :, 1] * y_ratio
        self.proposals[:, :, 2] = self.proposals[:, :, 2] * x_ratio
        self.proposals[:, :, 3] = self.proposals[:, :, 3] * y_ratio

        # proposals1_5 = proposals*1.5

        for image, i in zip(images, range(self.proposals.shape[0])):
            self.proposals[i][:, 0] = self.proposals[i][:, 0].clamp(
                0, image.shape[2]-2)
            self.proposals[i][:, 1] = self.proposals[i][:, 1].clamp(
                0, image.shape[1]-2)
            self.proposals[i][:, 2] = self.proposals[i][:, 2].clamp(
                0, image.shape[2]-2)
            self.proposals[i][:, 3] = self.proposals[i][:, 3].clamp(
                0, image.shape[1]-2)
        width = self.proposals[:, :, 3] - self.proposals[:, :, 1]
        height = self.proposals[:, :, 2] - self.proposals[:, :, 0]

        # 将小的矩形框进行调整
        for i in range(len(images)):
            for j in range(self.proposals.shape[1]):
                if(width[i, j] <= 0):
                    if(self.proposals[i, j, 3] <= 2):
                        self.proposals[i, j, 3] = self.proposals[i, j, 3]+2
                        self.proposals[i, j, 1] = self.proposals[i, j, 3]+1
                    else:
                        self.proposals[i, j, 3] = self.proposals[i, j, 3]-1
                        self.proposals[i, j, 1] = self.proposals[i, j, 3]-2
                if(height[i, j] <= 0):
                    if(self.proposals[i, j, 2] <= 2):
                        self.proposals[i, j, 2] = self.proposals[i, j, 2]+2
                        self.proposals[i, j, 0] = self.proposals[i, j, 2]+1
                    else:
                        self.proposals[i, j, 2] = self.proposals[i, j, 2]-1
                        self.proposals[i, j, 0] = self.proposals[i, j, 2]-2

        # 将变换后的proposals输入clip获得50x512的image_embedding
        postprocess = transforms.ToPILImage()
        self.proposals = self.proposals.long()
        image_embeddings = torch.empty(
            len(images), self.proposals.shape[1], 512)

        for image, j in zip(images, range(len(images))):
            for i in range(self.proposals.shape[1]):
                image_input = self.preprocess(postprocess(
                    image[:, self.proposals[j, i, 1]:self.proposals[j, i, 3], self.proposals[j, i, 0]:self.proposals[j, i, 2]])).unsqueeze(0).to(device)
                image_embedding = self.model.encode_image(image_input)
                image_embeddings[j, i] = image_embedding

        # 归一化
        v = torch.linalg.norm(image_embeddings, ord=1,
                              dim=2).unsqueeze(2).repeat([1, 1, 512])
        image_embeddings = image_embeddings / v

        return image_embeddings

    def get_label(self, proposal, target, image):
        """
        计算每个proposal的标签
        """
        text_label = torch.zeros([1, 10])
        for i in range(len(target['labels'])):
            if nms.iot(proposal.to('cpu'), target['boxes'][i]) >= 0.3:
                text_label[0, target['labels'][i]] = 1
        return text_label

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

    def sim(self, a, b):
        return (a @ b.T)/(torch.linalg.norm(a, ord=1)*torch.linalg.norm(b, ord=1))

    def Loss_text(self, text_embedding, region_embedding, targets):
        print(region_embedding.shape)
        print(text_embedding.shape)
        losses = 0
        for regions in region_embedding:
            for region in regions:
                Zr = self.sim(region.unsqueeze(0), text_embedding.float())
                loss_t = torch.nn.functional.cross_entropy(
                    torch.nn.functional.softmax(Zr/0.1, dim=1),)
                losses += loss_t

    def forward(self, images, targets):
        # backbone已经训练好
        self.backbone.eval().to(device)
        # backbone输入应该为image数组
        detection = self.backbone(images)
        self.backbone.to('cpu')
        region_embedding = self.region(
            self.backbone.roi_heads.box_head.features).reshape(2, int(len(self.backbone.roi_heads.box_head.features)/len(images)), 512)
        # 训练或者推理时都需要计算text_embedding
        text_embedding = self.VILD_text()
        if self.training:
            self.proposals = torch.stack(
                [proposal for proposal in self.backbone.proposals])
            images_embedding = self.VILD_image(images)
            self.Loss_text(text_embedding, region_embedding, targets)

            proposal_labels = torch.empty(
                [self.proposals.shape[0], self.proposals.shape[1], 10])
                
            #获取每个proposal对应的one-hot标签
            for i in range(self.proposals.shape[0]):
                for p, j in zip(self.proposals[i], self.proposals.shape[1]):
                    proposal_labels[i, j] = self.get_label(
                        p, targets[i], images[i])
            return self.proposals, region_embedding

        return region_embedding


if __name__ == '__main__':
    preprocess = vocDataset.get_transform(False)  # 主要作用是将图片数据转成tensor传入显存
    postprocess = transforms.ToPILImage()

    dataset = vocDataset.vocData(
        "/home/llrt/文档/VOCdevkit/VOC2012", transform=preprocess)
    model0 = VILD()
    model0.train()

    image0, target0 = dataset[15]
    image1, target1 = dataset[16]

    img = []
    tgt = []
    tgt.append(target0)
    tgt.append(target1)
    img.append(image0.to(device))
    img.append(image1.to(device))
    model0(img, tgt)

    # model0.backbone.roi_heads.box_predictor.box
    print(tgt)
    # image = VILD_image()
    # print(image.backbone.proposals)
